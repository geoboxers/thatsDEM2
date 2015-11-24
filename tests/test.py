# -*- coding: utf-8 -*-
"""
Unittests for thatsDEM
@author: simlk
"""

import os
import sys
import unittest
import time
import logging
import numpy as np
HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, HOME)
from thatsDEM import pointcloud, triangle, array_geometry

LOG = logging.getLogger(__name__)

class Test(unittest.TestCase):

    def test_triangle(self):
        n1 = 1000
        n2 = 1000
        points = np.random.rand(n1, 2) * 1000.0
        z = np.random.rand(n1) * 100
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        LOG.info("Span of 'pointcloud': %.2f,%.2f,%.2f,%.2f" % (xmin, ymin, xmax, ymax))
        dx = (xmax - xmin)
        dy = (ymax - ymin)
        cx, cy = points.mean(axis=0)
        xy = np.random.rand(n2, 2) * [dx, dy] * 0.3 + [cx, cy]
        t1 = time.clock()
        tri = triangle.Triangulation(points, -1)
        t2 = time.clock()
        t3 = t2 - t1
        LOG.info("Building triangulation and index of %d points: %.4f s" % (n1, t3))
        LOG.info(tri.inspect_index())
        t1 = time.clock()
        tri.optimize_index()
        t2 = time.clock()
        t3 = t2 - t1
        LOG.info("\n%s\nOptimizing index: %.4fs" % ("*" * 50, t3))
        LOG.info(tri.inspect_index())
        t1 = time.clock()
        T = tri.find_triangles(xy)
        t2 = time.clock()
        t3 = t2 - t1
        LOG.info("Finding %d simplices: %.4f s, pr. 1e6: %.4f s" % (n2, t3, t3 / n2 * 1e6))
        self.assertGreaterEqual(T.min(), 0)
        self.assertLess(T.max(), tri.ntrig)
        t1 = time.clock()
        zi = tri.interpolate(z, points)
        t2 = time.clock()
        t3 = t2 - t1
        LOG.info("Interpolation test of vertices:  %.4f s, pr. 1e6: %.4f s" % (t3, t3 / n1 * 1e6))
        D = np.fabs(z - zi)
        LOG.info("Diff: %.15g, %.15g, %.15g" % (D.max(), D.min(), D.mean()))
        self.assertLess(D.max() , 1e-4)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
