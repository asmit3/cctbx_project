from __future__ import division
import mmtbx.monomer_library.pdb_interpretation
import iotbx.mtz
from cctbx.array_family import flex
import time
from mmtbx import monomer_library
import mmtbx.refinement.real_space.fit_residues
import scitbx.math
import mmtbx.idealized_aa_residues.rotamer_manager

pdb_answer = """\
CRYST1   14.074   16.834   17.360  90.00  90.00  90.00 P 1
ATOM      1  N   ARG A  21       8.318  11.834   9.960  1.00 10.00           N
ATOM      2  CA  ARG A  21       7.171  11.092   9.451  1.00 10.00           C
ATOM      3  C   ARG A  21       6.012  11.120  10.440  1.00 10.00           C
ATOM      4  O   ARG A  21       5.017  10.416  10.266  1.00 10.00           O
ATOM      5  CB  ARG A  21       7.564   9.645   9.143  1.00 10.00           C
ATOM      6  CG  ARG A  21       8.560   9.501   8.003  1.00 10.00           C
ATOM      7  CD  ARG A  21       8.125  10.300   6.785  1.00 10.00           C
ATOM      8  NE  ARG A  21       8.926   9.982   5.607  1.00 10.00           N
ATOM      9  CZ  ARG A  21       8.893  10.674   4.473  1.00 10.00           C
ATOM     10  NH1 ARG A  21       8.093  11.727   4.359  1.00 10.00           N
ATOM     11  NH2 ARG A  21       9.655  10.313   3.451  1.00 10.00           N
END
"""

pdb_for_map = """
ATOM      1  O   HOH S   0       8.318  11.834   9.960  1.00 10.00           O
ATOM      2  O   HOH S   1       7.171  11.092   9.451  1.00 10.00           O
ATOM      3  O   HOH S   2       6.012  11.120  10.440  1.00 10.00           O
ATOM      4  O   HOH S   3       5.017  10.416  10.266  1.00 10.00           O
ATOM      5  O   HOH S   4       7.564   9.645   9.143  1.00 10.00           O
ATOM      6  O   HOH S   5       8.560   9.501   8.003  1.00 10.00           O
ATOM      7  O   HOH S   6       8.125  10.300   6.785  1.00 10.00           O
ATOM      8  O   HOH S   7       8.926   9.982   5.607  1.00 10.00           O
ATOM      9  O   HOH S   8       8.893  10.674   4.473  1.00 10.00           O
ATOM     10  O   HOH S   9       8.093  11.727   4.359  1.00 10.00           O
ATOM     11  O   HOH S  10       9.655  10.313   3.451  1.00 10.00           O
TER
"""

pdb_poor = """\
CRYST1   14.074   16.834   17.360  90.00  90.00  90.00 P 1
ATOM      1  N   ARG A  21       8.318  11.834   9.960  1.00 10.00           N
ATOM      2  CA  ARG A  21       7.248  10.924   9.570  1.00 10.00           C
ATOM      3  C   ARG A  21       6.012  11.120  10.440  1.00 10.00           C
ATOM      4  O   ARG A  21       5.064  10.337  10.375  1.00 10.00           O
ATOM      5  CB  ARG A  21       7.724   9.472   9.652  1.00 10.00           C
ATOM      6  CG  ARG A  21       8.797   9.112   8.637  1.00 10.00           C
ATOM      7  CD  ARG A  21       9.187   7.647   8.741  1.00 10.00           C
ATOM      8  NE  ARG A  21      10.266   7.301   7.820  1.00 10.00           N
ATOM      9  CZ  ARG A  21      10.871   6.118   7.790  1.00 10.00           C
ATOM     10  NH1 ARG A  21      10.505   5.162   8.634  1.00 10.00           N
ATOM     11  NH2 ARG A  21      11.844   5.891   6.920  1.00 10.00           N
TER
ATOM      1  O   HOH S   0       8.318  11.834   9.960  1.00 10.00           O
ATOM      2  O   HOH S   1       7.171  11.092   9.451  1.00 10.00           O
ATOM      3  O   HOH S   2       6.012  11.120  10.440  1.00 10.00           O
ATOM      4  O   HOH S   3       5.017  10.416  10.266  1.00 10.00           O
ATOM      5  O   HOH S   4       7.564   9.645   9.143  1.00 10.00           O
ATOM      6  O   HOH S   5       8.560   9.501   8.003  1.00 10.00           O
ATOM      7  O   HOH S   6       8.125  10.300   6.785  1.00 10.00           O
ATOM      8  O   HOH S   7       8.926   9.982   5.607  1.00 10.00           O
ATOM      9  O   HOH S   8       8.893  10.674   4.473  1.00 10.00           O
ATOM     10  O   HOH S   9       8.093  11.727   4.359  1.00 10.00           O
ATOM     11  O   HOH S  10       9.655  10.313   3.451  1.00 10.00           O
TER
END
"""

def exercise(rotamer_manager, sin_cos_table, d_min = 1.0,
             resolution_factor = 0.1):
  # Make sure it kicks off existing water. Simple case: no alternatives.
  #
  # answer PDB
  pdb_inp = iotbx.pdb.input(source_info=None, lines=pdb_for_map)
  pdb_inp.write_pdb_file(file_name = "for_map.pdb")
  xrs_answer = pdb_inp.xray_structure_simple()
  # answer map
  pdb_inp = iotbx.pdb.input(source_info=None, lines=pdb_answer)
  xrs_map = pdb_inp.xray_structure_simple()
  f_calc = xrs_map.structure_factors(d_min = d_min).f_calc()
  fft_map = f_calc.fft_map(resolution_factor=resolution_factor)
  fft_map.apply_sigma_scaling()
  target_map = fft_map.real_map_unpadded()
  mtz_dataset = f_calc.as_mtz_dataset(column_root_label = "FCmap")
  mtz_object = mtz_dataset.mtz_object()
  mtz_object.write(file_name = "answer.mtz")
  # poor
  mon_lib_srv = monomer_library.server.server()
  processed_pdb_file = monomer_library.pdb_interpretation.process(
    mon_lib_srv              = mon_lib_srv,
    ener_lib                 = monomer_library.server.ener_lib(),
    raw_records              = flex.std_string(pdb_poor.splitlines()),
    strict_conflict_handling = True,
    force_symmetry           = True,
    log                      = None)
  pdb_hierarchy_poor = processed_pdb_file.all_chain_proxies.pdb_hierarchy
  xrs_poor = processed_pdb_file.xray_structure()
  sites_cart_poor = xrs_poor.sites_cart()
  pdb_hierarchy_poor.write_pdb_file(file_name = "poor.pdb")
  #
  result = mmtbx.refinement.real_space.fit_residues.run(
    pdb_hierarchy     = pdb_hierarchy_poor,
    crystal_symmetry  = xrs_poor.crystal_symmetry(),
    map_data          = target_map,
    do_all            = True,
    rotamer_manager   = rotamer_manager,
    sin_cos_table     = sin_cos_table,
    mon_lib_srv       = mon_lib_srv)
  result.pdb_hierarchy.write_pdb_file(file_name = "refined.pdb")
  ###
  sel = result.pdb_hierarchy.atom_selection_cache().selection("not water")
  result_hierarchy = result.pdb_hierarchy.select(sel)

  pdb_inp = iotbx.pdb.input(source_info=None, lines=pdb_answer)
  pdb_inp.write_pdb_file(file_name = "answer.pdb")
  xrs_answer = pdb_inp.xray_structure_simple()
  dist = flex.max(flex.sqrt((xrs_answer.sites_cart() -
    result_hierarchy.atoms().extract_xyz()).dot()))
  print dist
  assert dist < 0.4, dist

if(__name__ == "__main__"):
  t0 = time.time()
  # load rotamer manager
  rotamer_manager = mmtbx.idealized_aa_residues.rotamer_manager.load()
  # pre-compute sin and cos tables
  sin_cos_table = scitbx.math.sin_cos_table(n=10000)
  exercise(
    rotamer_manager = rotamer_manager,
    sin_cos_table   = sin_cos_table)
  print "Time: %6.4f"%(time.time()-t0)
