# -*- coding: utf-8 -*-
"""
Unittests for pointcloud
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

class TestTriangle(unittest.TestCase):

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

class TestPointcloud(unittest.TestCase):

    def test_pointcloud_constructor1(self):
        LOG.info("Testing pointcloud constructor")
        pc = pointcloud.Pointcloud(np.ones((2,2)), np.ones(2), some_attr = np.ones(2, dtype=np.uint8))
        self.assertIn("some_attr", pc.attributes)
        self.assertTrue(pc.some_attr.dtype == np.uint8)
    
    def test_pointcloud_constructor_bad(self):
        LOG.info("Testing pointcloud constructor -bad")
        with self.assertRaises(AssertionError):
            pc = pointcloud.Pointcloud(np.ones((2,2)), np.ones(2), some_attr = np.ones(4, dtype=np.uint8))
        
    def test_pointcloud_empty_like(self):
        LOG.info("Testing pointcloud empty_like factory function")
        pc = pointcloud.Pointcloud(np.ones((2,2)), np.ones(2), some_attr = np.ones(2))
        empty = pointcloud.empty_like(pc)
        self.assertSetEqual(pc.attributes, empty.attributes)
        self.assertEqual(empty.size, 0)
    
    def test_extend_pointcloud1(self):
        LOG.info("Testing pointcloud extension - bad")
        pc1 = pointcloud.Pointcloud(np.ones((2,2)), np.ones(2), some_attr = np.ones(2))
        pc2 = pointcloud.Pointcloud(np.ones((2,2)), np.ones(2))
        with self.assertRaises(ValueError):
            pc1.extend(pc2)
    
    def test_extend_pointcloud2(self):
        LOG.info("Testing pointcloud extension - ok")
        pc1 = pointcloud.Pointcloud(np.ones((2,2)), np.ones(2), some_attr = np.ones(2))
        pc2 = pointcloud.Pointcloud(np.ones((2,2)), np.ones(2), some_attr = np.ones(2)*3)
        pc1.extend(pc2)
        self.assertEqual(pc1.size, 4)
        self.assertIn("some_attr",pc1.attributes)
    
    def test_extend_pointcloud3(self):
        LOG.info("Testing pointcloud extension - least common")
        pc1 = pointcloud.Pointcloud(np.ones((2,2)), np.ones(2), some_attr = np.ones(2), some_other= np.ones(2))
        pc2 = pointcloud.Pointcloud(np.ones((2,2)), np.ones(2), some_attr = np.ones(2)*3)
        pc1.extend(pc2, least_common=True)
        self.assertSetEqual(pc1.attributes,{"some_attr"})
    
    def test_thin_pointcloud(self):
        LOG.info("Testing thin pointcloud")
        pc = pointcloud.Pointcloud(np.ones((5,2)), np.ones(5), some_attr = np.ones(5), some_other= np.ones(5))
        M = np.array([1,0,1,1,0]).astype(np.bool)
        pc.thin(M)
        self.assertEqual(pc.size, M.sum())
        self.assertSetEqual(pc.attributes, {"some_attr", "some_other"})

    def test_cut_pointcloud(self):
        LOG.info("Testing cut poincloud")
        pc = pointcloud.Pointcloud(np.ones((5,2)), np.ones(5), some_attr = np.ones(5), some_other= np.ones(5))
        M = np.array([1,0,1,1,0]).astype(np.bool)
        pc = pc.cut(M)
        self.assertEqual(pc.size, M.sum())
        self.assertSetEqual(pc.attributes, {"some_attr", "some_other"})
    
    def test_lidar_pointcloud(self):
        LOG.info("Testing lidar pointcloud")
        pc = pointcloud.LidarPointcloud(np.ones((3,2)), np.ones(3), c=[2,2,3], some_attr = np.ones(3))
        self.assertItemsEqual(pc.get_classes(), [2,3])
        self.assertEqual(pc.cut_to_class(2).size, 2)
    
    def test_sort_pointcloud(self):
        LOG.info("Test pointcloud sorting")
        r = np.linspace(0, np.pi*2, 100)
        xy = np.column_stack((r*np.cos(r), r*np.sin(r)))*5
        c = np.arange(xy.shape[0])
        pc = pointcloud.Pointcloud(xy, np.ones(xy.shape[0]), c=c)
        pc.sort_spatially(1, keep_sorting=True)
        self.assertTrue((c != pc.c).any())
        pc.sort_back()
        self.assertTrue((pc.c == c).all())
    
    def test_pointcloud_might_overlap(self):
        LOG.info("Test pointcloud sorting")
        pc1= pointcloud.fromArray(np.ones((10,10)), [0, 1, 0, 10, 0, -1])
        pc2= pointcloud.fromArray(np.ones((10,10)), [0, 1, 0, 5, 0, -1])
        self.assertTrue(pc1.might_overlap(pc2))
        pc1.affine_transformation_2d(T=(30,30))
        self.assertFalse(pc1.might_overlap(pc2))
    
    def test_pointcloud_attributes(self):
        LOG.info("Test pointcloud attributes")
        pc= pointcloud.Pointcloud(np.ones((10,2)), np.ones(10), a=np.arange(10))
        with self.assertRaises(pointcloud.InvalidArrayError):
            pc.xy = 10
        with self.assertRaises(pointcloud.InvalidArrayError):
            pc.a = "abc"
        # Should be ok
        pc.a = range(10, 0, -1)
        self.assertEqual(pc.a[0],10)
    
    def test_pointcloud_min_filter(self):
        LOG.info("Test pointcloud min filter")
        pc = pointcloud.Pointcloud(((0,0),(1,0),(2,0),(3,0)), (1,2,3,4))
        pc.sort_spatially(2)
        z = pc.min_filter(1.5)
        self.assertTrue((z == (1,1,2,3)).all())
        
        
        
        
        
    
    
        
        
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
