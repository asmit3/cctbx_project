from __future__ import division
# LIBTBX_SET_DISPATCHER_NAME phenix.pdbtools

import mmtbx.pdbtools
import sys, os
from libtbx import runtime_utils
import mmtbx.utils
import iotbx.pdb
from iotbx.pdb import combine_unique_pdb_files, write_whole_pdb_file
from iotbx.phil import process_command_line_with_files
from iotbx.cif import write_whole_cif_file
from scitbx.array_family import flex
from libtbx import group_args
from libtbx.utils import check_if_output_directory_exists

master_params_str = """\
model_file_name = None
  .type = path
  .multiple = True
  .help = Model file name
  .short_caption = Input model
  .style = bold input_file file_type:pdb file_type_default
output
  .help = Write out file with modified model (file name is defined in \
          write_modified)
  .style = noauto
{
  file_name=None
    .type=path
    .input_size=400
    .short_caption = Output model file
    .help = Default is the original file name with the file extension \
            replaced by "_modified.pdb".
    .style = bold new_file file_type:pdb
  format = *pdb mmcif
    .type = choice
    .help = Choose the output format of coordinate file (PDB or mmCIF)
}
include scope mmtbx.pdbtools.master_params
include scope libtbx.phil.interface.tracking_params
"""

def master_params():
  return iotbx.phil.parse(master_params_str, process_includes=True)

def broadcast(m, log):
  print >> log, "-"*79
  print >> log, m
  print >> log, "*"*len(m)

def get_inputs(args, log, master_params):
  """
  Eventually, this will be centralized.
  """
  cmdline = process_command_line_with_files(
    args         = args,
    master_phil  = master_params,
    pdb_file_def = 'model_file_name'
  )
  params = cmdline.work.extract()
  # Model
  file_names = params.model_file_name
  pdb_combined = combine_unique_pdb_files(file_names = file_names)
  pdb_inp = iotbx.pdb.input(
    source_info = None,
    lines       = flex.std_string(pdb_combined.raw_records),
    raise_sorry_if_format_error = True)
  pdb_hierarchy = pdb_inp.construct_hierarchy()
  # Crystal symmetry
  fake_crystal_symmetry = False
  crystal_symmetry = pdb_inp.crystal_symmetry()
  if(crystal_symmetry is None or
     crystal_symmetry.is_empty() or
     crystal_symmetry.is_nonsence()):
    fake_crystal_symmetry = True
    from cctbx import uctbx
    crystal_symmetry = \
      uctbx.non_crystallographic_unit_cell_with_the_sites_in_its_center(
        sites_cart=pdb_inp.atoms().extract_xyz(),
        buffer_layer=5).crystal_symmetry()
  #
  xray_structure = pdb_hierarchy.extract_xray_structure(
    crystal_symmetry = crystal_symmetry)
  #
  return group_args(
    params                = params,
    pdb_file_names        = file_names,
    pdb_hierarchy         = pdb_hierarchy,
    xray_structure        = xray_structure,
    pdb_inp               = pdb_inp,
    fake_crystal_symmetry = fake_crystal_symmetry,
    crystal_symmetry      = crystal_symmetry)

def run(args, out=sys.stdout, replace_stderr=True):
  log = mmtbx.utils.set_log(args, out=out, replace_stderr=replace_stderr)
  broadcast(m="phenix.pdbtools tools for PDB model manipulations.", log=log)
  if(len(args)==0):
    master_params().show(out = log)
    return None
  inputs = get_inputs(args=args, log=log, master_params=master_params())
  ### get i/o file names
  ofn = inputs.params.output.file_name
  output_format = inputs.params.output.format
  ifn = inputs.pdb_file_names
  if(ofn is None):
    if output_format == "pdb": ext = "pdb"
    elif output_format == "mmcif": ext = "cif"
    if(len(ifn)==1): ofn = os.path.basename(ifn[0]) + "_modified."+ext
    elif(len(ifn)>1): ofn = os.path.basename(ifn[0]) + "_et_al_modified"+ext
    else:
      pdbout = os.path.basename(inputs.pdb_file_names[0])
      ofn = pdbout+"_modified."+ext
  # Show parameters
  broadcast(m="Complete set of parameters", log=log)
  master_params().format(inputs.params).show(out = log)
  print >> log
  # Run calcs
  broadcast(m="Performing manipulations", log=log)
  task_obj = mmtbx.pdbtools.modify(
    xray_structure = inputs.xray_structure,
    params         = inputs.params.modify,
    pdb_hierarchy  = inputs.pdb_hierarchy,
    log            = log)
  results = task_obj.get_results()
  #
  broadcast(m="Writing output model", log=log)
  print >> log, "Output model file name: ", ofn
  if(inputs.fake_crystal_symmetry):
    crystal_symmetry = None
  else:
    crystal_symmetry = results.crystal_symmetry
  if output_format == "pdb":
    write_whole_pdb_file(
      file_name        = ofn,
      pdb_hierarchy    = results.pdb_hierarchy,
      ss_annotation    = inputs.pdb_inp.extract_secondary_structure(),
      crystal_symmetry = crystal_symmetry,
      append_end       = True,
      atoms_reset_serial_first_value = 1)
  elif output_format =="mmcif":
    write_whole_cif_file(
      file_name        = ofn,
      pdb_hierarchy    = results.pdb_hierarchy,
      ss_annotation    = inputs.pdb_inp.extract_secondary_structure(),
      crystal_symmetry = crystal_symmetry)
  broadcast(m="All done.", log=log)
  return None

def validate_params (params, callback=None) :
  if (params.model_file_name is None) :
    raise Sorry("No PDB file(s) specified.")
  elif (params.output.file_name is not None) :
    if (os.path.isdir(params.output.file_name)) :
      raise Sorry("The specified output file is a currently existing "+
                  "directory.")
    check_if_output_directory_exists(params.output.file_name)
  return True

class launcher (runtime_utils.target_with_save_result) :
  def run (self) :
    return run(args=list(self.args), out=sys.stdout)

if (__name__ == "__main__"):
  assert run(args=sys.argv[1:]) is None # assert here is intentional
