import sys
import math
import random
from cctbx_boost.arraytbx import shared
from cctbx_boost import sgtbx
from cctbx_boost import sftbx
from cctbx import xutils
from cctbx.development import debug_utils

def OneCycle(settings):
  print "Testing Miller index mappings"
  for LookupSymbol in settings:
    SgSymbols = sgtbx.SpaceGroupSymbols(LookupSymbol)
    HSym = SgSymbols.Hall()
    SgOps = sgtbx.SpaceGroup(HSym)
    SgInfo = SgOps.Info()
    print "SpaceGroup %s (%d) %s" % (
      SgInfo.BuildLookupSymbol(),
      SgInfo.SgNumber(),
      SgInfo.BuildHallSymbol())
    sys.stdout.flush()
    UnitCell = debug_utils.get_compatible_unit_cell(SgInfo, 1000).UnitCell
    asu = sgtbx.ReciprocalSpaceASU(SgInfo)
    for friedel_flag in (1,0):
      miller_indices = sftbx.BuildMillerIndices(
        UnitCell, SgInfo, friedel_flag, 2.)
      for h_asym in miller_indices:
        h_seq = SgOps.getEquivMillerIndices(h_asym)
        # exercise class PhaseInfo
        restr = h_seq.getPhaseRestriction()
        assert not restr.SysAbsWasTested()
        if (h_seq.isCentric()):
          phi_asym = restr.HT_angle() + random.choice((0,1)) * math.pi
        else:
          phi_asym = random.random() * 2 * math.pi
        phi_asym = (phi_asym, phi_asym * 180 / math.pi)
        for deg in (0,1):
          assert restr.isValidPhase(phi_asym[deg], deg)
        if (h_seq.isCentric()):
          assert not restr.isValidPhase(phi_asym[0] + math.pi / 180)
          assert not restr.isValidPhase(phi_asym[1] - 1, 1)
        # exercise class Miller_SymEquivIndices
        f_asym = xutils.ampl_phase_as_f((random.random(), phi_asym[0]))
        hlc_asym = [random.random() for i in xrange(4)]
        for i_eq in xrange(h_seq.M(friedel_flag)):
          h_eq = h_seq(i_eq)
          for deg in (0,1):
            phi_eq = h_eq.phase_eq(phi_asym[deg])
            phi_in = h_eq.phase_in(phi_eq)
            assert abs(phi_in - phi_asym[deg]) < 1.e-5
          f_eq = h_eq.complex_eq(f_asym)
          f_in = h_eq.complex_in(f_eq)
          a, p = xutils.f_as_ampl_phase(f_in)
          assert abs(p - phi_asym[0]) < 1.e-5
          hlc_eq = h_eq.hl_eq(hlc_asym)
          hlc_in = h_eq.hl_in(hlc_eq)
          for i in xrange(4):
            assert abs(hlc_in[i] - hlc_asym[i]) < 1.e-5
      # exercise expand_to_p1, AsymIndex, IndexTableLayoutAdapter
      p1_miller_indices = shared.Miller_Index()
      sgtbx.expand_to_p1(
        SgOps, friedel_flag, miller_indices, p1_miller_indices)
      h_dict = {}
      for i in xrange(miller_indices.size()):
        h_dict[miller_indices[i]] = 0
      for p1_h in p1_miller_indices:
        h_asu = sgtbx.Miller_AsymIndex(SgOps, asu, p1_h)
        if (friedel_flag):
          h_dict[h_asu.HermitianLayout().H()] += 1
        else:
          h_dict[h_asu.AnomalousLayout().H()] += 1
      f = 1
      if (friedel_flag): f = 2
      for h, c in h_dict.items():
        h_seq = SgOps.getEquivMillerIndices(h)
        assert f * c == h_seq.M(friedel_flag)
        assert c == h_seq.n_p1_listing(friedel_flag)

def run(timing=1):
  from tst import run_other
  run_other(sys.argv[1:], timing, OneCycle,
    ("P 3 1 m",))

if (__name__ == "__main__"):
  run()
