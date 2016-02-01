# -*- coding: utf-8 -*-
"""
Unittests for various thatsDEM2 modules
@author: simlk
"""

import os
import sys
import unittest
import time
import logging
import numpy as np

from thatsDEM2 import array_geometry, grid

LOG = logging.getLogger(__name__)


class OtherTests(unittest.TestCase):

    def test_curve_simplify(self):
        LOG.info("Trying array_geometry.simplify_linestring")
        xy = np.asarray([(0.0, 0.0), (0.0, 1.0), (1.1, 1.1)], dtype=np.float64)
        xy_out = array_geometry.simplify_linestring(xy, 1/np.sqrt(2) + 0.1)
        self.assertEqual(xy_out.shape[0], 2)

    def test_grid_walk_mask(self):
        LOG.info("Testing grid.walk_mask")
        M = np.zeros((6,6), dtype=np.bool)
        M[0, :] = 1
        M[:3, 0] = 1
        M[:, 5] = 1
        M[5, :] = 1
        path = grid.walk_mask(M, (0, 0), (5, 0))
        self.assertEqual(path.shape[0], 14)
  

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
