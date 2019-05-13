from __future__ import division
from __future__ import print_function
from libtbx.utils import format_cpu_times, Sorry
import mmtbx

def exercise():
  r = mmtbx.map_names(map_name_string = "anom")
  assert r.format() == "anomalous_difference", r.format()
  r = mmtbx.map_names(map_name_string = "anomalous")
  assert r.format() == "anomalous_difference", r.format()
  r = mmtbx.map_names(map_name_string = " Anomal-diff  ")
  assert r.format() == "anomalous_difference", r.format()
  r = mmtbx.map_names(map_name_string = "LLG")
  assert r.format() == "phaser_sad_llg"
  r = mmtbx.map_names(map_name_string = " SAD")
  assert r.format() == "phaser_sad_llg"
  #
  r = mmtbx.map_names(map_name_string = "3.mFo-2DFc ")
  assert r.format() == "3mFobs-2DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "mFo-DFc ")
  assert r.format() == "mFobs-DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "mFo-1.DFc ")
  assert r.format() == "mFobs-DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "+3.mFo+2DFc ")
  assert r.format() == "3mFobs+2DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "-3.mFo+2DFc")
  assert r.format() == "-3mFobs+2DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "-2DFc+3.mFo")
  assert r.format() == "3mFobs-2DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "- 2 D F c + 3 . m F o ")
  assert r.format() == "3mFobs-2DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "-  D F ca lc 2.0 +  m F oBS 3 .0")
  assert r.format() == "3mFobs-2DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "-2DFc-3.mFo")
  assert r.format() == "-3mFobs-2DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "+2DFc+3.mFo")
  assert r.format() == "3mFobs+2DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "-3.mF_OBS-2D_F_cAlC")
  assert r.format() == "-3mFobs-2DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "+3.mF_o+2 . D F_MODEL")
  assert r.format() == "3mFobs+2DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "0.75mFobs-1.37DFmodel")
  assert r.format() == "0.75mFobs-1.37DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "0.99999mFobs-1.00001DFmodel")
  assert r.format() == "0.99999mFobs-1.00001DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "0mFobs-0DFmodel")
  assert r.format() == "0mFobs-0DFmodel", r.format()
  #
  r = mmtbx.map_names(map_name_string = "3.Fo-2Fc ")
  assert r.format() == "3Fobs-2Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "Fo-Fc ")
  assert r.format() == "Fobs-Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "Fo-1.Fc ")
  assert r.format() == "Fobs-Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "+3.Fo+2Fc ")
  assert r.format() == "3Fobs+2Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "-3.Fo+2Fc")
  assert r.format() == "-3Fobs+2Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "-2Fc+3.Fo")
  assert r.format() == "3Fobs-2Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "- 2  F c + 3 .  F o ")
  assert r.format() == "3Fobs-2Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "-   F ca lc 2.0 +   F oBS 3 .0")
  assert r.format() == "3Fobs-2Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "-2Fc-3.Fo")
  assert r.format() == "-3Fobs-2Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "+2Fc+3.Fo")
  assert r.format() == "3Fobs+2Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "-3.F_OBS-2_F_cAlC")
  assert r.format() == "-3Fobs-2Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "+3.F_o+2 .  F_MODEL")
  assert r.format() == "3Fobs+2Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "0.75Fobs-1.37Fmodel")
  assert r.format() == "0.75Fobs-1.37Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "0.99999Fobs-1.00001Fmodel")
  assert r.format() == "0.99999Fobs-1.00001Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "0Fobs-0Fmodel")
  assert r.format() == "0Fobs-0Fmodel", r.format()
  #
  r = mmtbx.map_names(map_name_string = "fc")
  assert r.format() == "0Fobs+Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "FMODEL")
  assert r.format() == "0Fobs+Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "dfc")
  assert r.format() == "0mFobs+DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "dFMODEL")
  assert r.format() == "0mFobs+DFmodel", r.format()
  #
  r = mmtbx.map_names(map_name_string = "fo")
  assert r.format() == "Fobs-0Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "FoBS")
  assert r.format() == "Fobs-0Fmodel", r.format()
  r = mmtbx.map_names(map_name_string = "mfo")
  assert r.format() == "mFobs-0DFmodel", r.format()
  r = mmtbx.map_names(map_name_string = "MFOBS")
  assert r.format() == "mFobs-0DFmodel", r.format()
  #
  r = mmtbx.map_names(map_name_string = "3.mFo-2DFc__filled")
  assert r.format() == "3mFobs-2DFmodel_filled", r.format()
  r = mmtbx.map_names(map_name_string = "3.Fo-2Fc+filled")
  assert r.format() == "3Fobs-2Fmodel_filled", r.format()
  #
  def check_expected_error(s):
    cntr = 0
    try: r = mmtbx.map_names(map_name_string = s)
    except Sorry as e:
      assert str(e).count("Wrong map type requested: %s"%s)==1
      cntr += 1
    assert cntr == 1
  #
  check_expected_error("2mFoDFc")
  check_expected_error("fofc")
  check_expected_error("2mFo*DFc")
  check_expected_error("2mFo/DFc")
  check_expected_error("2mFo:DFc")
  check_expected_error("2mFo_DFc")
  check_expected_error("2mFo-DC")
  check_expected_error("2mo-DC")
  check_expected_error("2mFo-Fc")
  check_expected_error("2Fo-DFc")
  check_expected_error("2DFo-mFc")
  check_expected_error("2DFo-DFc")
  check_expected_error("2mFo-mFc")

def run():
  exercise()

if (__name__ == "__main__"):
  run()
  print(format_cpu_times())
