# -*- coding: utf-8 -*-
"""
Unittests for triangle
@author: simlk
"""

import os
import sys
import unittest
import time
import logging
import numpy as np
from thatsDEM import triangle

class TestTriangle(unittest.TestCase):
    
    def setUp(self):
        self.n1 = 4000
        self.n2 = 4000
        self.points = np.random.rand(self.n1, 2) * 1000.0
        t1 = time.clock()
        self.tri = triangle.Triangulation(self.points, -1)
        t2 = time.clock()
        t3 = t2 - t1
        LOG.info("Building triangulation and index of %d points: %.4f s" % (self.n1, t3))
    
    def test_optimize_index(self):
        LOG.info("Testing optimize index.")
        self.tri.optimize_index()
    
    def test_find_triangles(self):
        LOG.info("Testing Find triangles.")
        xmin, ymin = self.points.min(axis=0)
        xmax, ymax = self.points.max(axis=0)
        dx = (xmax - xmin)
        dy = (ymax - ymin)
        cx, cy = self.points.mean(axis=0)
        xy = np.random.rand(self.n2, 2) * [dx, dy] * 0.3 + [cx, cy]
        t1 = time.time()
        T = self.tri.find_triangles(xy)
        t2 = time.time()
        t3 = t2 - t1
        LOG.info("Finding %d simplices: %.4f s, pr. 1e6: %.4f s" % (self.n2, t3, t3 / self.n2 * 1e6))
        self.assertGreaterEqual(T.min(), 0)
        self.assertLess(T.max(), self.tri.ntrig)
    
    def test_interpolation(self):
        z = np.random.rand(self.n1) * 100
        t1 = time.clock()
        zi = self.tri.interpolate(z, self.points)
        t2 = time.clock()
        t3 = t2 - t1
        LOG.info("Interpolation test of vertices:  %.4f s, pr. 1e6: %.4f s" % (t3, t3 / self.n1 * 1e6))
        D = np.fabs(z - zi)
        self.assertLess(D.max() , 1e-4)

LOG = logging.getLogger(__name__)

        
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
