from __future__ import division
import cctbx.array_family.flex # import dependency

import boost.python
ext = boost.python.import_ext("mmtbx_bulk_solvent_ext")
from mmtbx_bulk_solvent_ext import *
from multi_mask_bulk_solvent import multi_mask_bulk_solvent # special import
