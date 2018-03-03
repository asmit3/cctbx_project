from __future__ import absolute_import, division
from dxtbx.format.Format import Format
from dxtbx.format.FormatStill import FormatStill
from dxtbx.format.FormatMultiImage import FormatMultiImage
from libtbx.phil import parse

locator_scope = parse("""
  data_source = None
    .type = str
    .help = file format as specified at LCLS,eg. exp=mfxo1916:run=20:smd\
           More info at https://confluence.slac.stanford.edu/display/PSDM/Manual#Manual-Datasetspecification
  detector_address = None
    .type = str
    .help = detector used for collecting the data at LCLS
""")

class FormatXTC(FormatMultiImage,FormatStill,Format):

  def __init__(self, image_file, **kwargs):
    from dxtbx import IncorrectFormatError
    if not self.understand(image_file):
      raise IncorrectFormatError(self, image_file)
    FormatMultiImage.__init__(self, **kwargs)
    FormatStill.__init__(self, image_file, **kwargs)
    Format.__init__(self, image_file, **kwargs)
    self._initialized=True

  @staticmethod
  def understand(image_file):
    ''' Extracts the datasource and detector_address from the image_file and then feeds it to PSANA
        If PSANA fails to read it, then input may not be an xtc/smd file. If success, then OK.
        If detector_address is not provided, a command line promp will try to get the address
        from the user '''
    try:
      from psana import DataSource, DetNames
    except ImportError:
      return False
    try:
      params = FormatXTC.params_from_phil(image_file)
    except Exception:
      return False
    if params.data_source is None:
      return False
    else:
      FormatXTC._img = params.data_source
      FormatXTC._src = params.detector_address

    ds = DataSource(FormatXTC._img)
    if FormatXTC._src is None:
      print 'This is an XTC file and can be read by PSANA'
      print 'Listed Below are the detector names associated with the experiment'
      names = DetNames('detectors')
      headers = ['Full Name','DAQ Alias','User Alias']
      maxlen = [len(h) for h in headers]
      for ntuple in names:
        lengths = [len(n) for n in ntuple]
        maxlen = [max(oldmax,length) for oldmax,length in zip(maxlen,lengths)]
      template = "{0:%d} | {1:%d} | {2:%d}" % tuple(maxlen)
      header = template.format("Full Name", "     DAQ Alias", "User Alias")
      print '-'*len(header)
      print header
      print '-'*len(header)
      for i, n in enumerate(names):
        print '%3d'%(i+1) + ') '+template.format(*n)
      print '-'*len(header)

      FormatXTC._src = names[int(raw_input("Please Enter name of detector numbered 1 through %d : "%(len(names))))-1][0]
    return True

  @staticmethod
  def params_from_phil(image_file):
    try:
      user_input = parse(file_name = image_file)
      working_phil = locator_scope.fetch(sources = [user_input])
      params = working_phil.extract()
      return params
    except Exception:
      return None

  def _get_datasource(self, image_file):
    from psana import DataSource

    params = self.params_from_phil(image_file)
    if params.data_source is None:
      return False
    else:
      img = params.data_source
    return DataSource(img)

if __name__ == '__main__':
  import sys
  for arg in sys.argv[1:]:
    print FormatXTC.understand(arg)
