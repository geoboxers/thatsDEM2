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
from thatsDEM2 import pointcloud

LOG = logging.getLogger(__name__)

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
    
    def _test_pointcloud_grid_filter(self, method, mean_val):
        LOG.info("Test pointcloud gridding, method: %s" % str(method))
        pc = pointcloud.fromArray(np.arange(100).reshape((10,10)), [0, 1, 0, 10, 0, -1])
        pc.sort_spatially(2)
        g = pc.get_grid(ncols=10, nrows=10, x1=0, x2=10, y1=0, y2=10, attr="z", srad=2, method=method)
        self.assertEqual(g.shape, (10,10))
        self.assertAlmostEqual(g.grid.mean(), mean_val, 3)
        
    def test_pointcloud_grid_idw_filter(self):
        self._test_pointcloud_grid_filter("idw_filter", 49.5)
    
    def test_pointcloud_grid_var_filter(self):
        self._test_pointcloud_grid_filter("var_filter", 96.434)
    
    def test_pointcloud_grid_mean_filter(self):
        self._test_pointcloud_grid_filter("mean_filter", 49.5)
    
    def test_pointcloud_grid_min_filter(self):
        self._test_pointcloud_grid_filter("min_filter", 32.24)
    
    def test_pointcloud_grid_max_filter(self):
        self._test_pointcloud_grid_filter("max_filter", 66.76)
    
    def test_pointcloud_grid_median_filter(self):
        self._test_pointcloud_grid_filter("median_filter", 49.5)
    
    def test_pointcloud_grid_cellcount(self):
        LOG.info("Test pointcloud gridding, method: cellcount")
        pc = pointcloud.fromArray(np.arange(100).reshape((10,10)), [0, 1, 0, 10, 0, -1])
        g = pc.get_grid(ncols=10, nrows=10, x1=0, x2=10, y1=0, y2=10, srad=2, method="cellcount")
        self.assertTrue((g.grid == 1).all())
    
    def test_pointcloud_grid_density_filter(self):
        LOG.info("Test pointcloud gridding, method: density_filter")
        pc = pointcloud.fromArray(np.arange(100).reshape((10,10)), [0, 1, 0, 10, 0, -1])
        pc.sort_spatially(2)
        g = pc.get_grid(ncols=10, nrows=10, x1=0, x2=10, y1=0, y2=10, srad=2, method="density_filter")
        self.assertGreater(g.grid.min(), 0.4)
        self.assertLess(g.grid.max(), 1.1)
        self.assertTrue((g.grid[:,0]==g.grid[:,-1]).all())
        self.assertTrue((g.grid[0,:]==g.grid[-1,:]).all())
    
    def test_pointcloud_grid_by_function(self):
        LOG.info("Test pointcloud gridding, method: np.max")
        pc = pointcloud.fromArray(np.arange(100).reshape((10,10)), [0, 1, 0, 10, 0, -1])
        g = pc.get_grid(ncols=2, nrows=2, x1=0, x2=10, y1=0, y2=10, method=np.max)
        self.assertTrue( (g.grid == np.array(((44., 49.), (94., 99.)))).all())
    
    def test_pointcloud_grid_most_frequent(self):
        LOG.info("Test pointcloud gridding, method: most_frequent")
        pc = pointcloud.fromArray(np.ones((10,10)), [0, 1, 0, 10, 0, -1])
        c = (np.arange(100) % 10).astype(np.int32)
        pc.set_attribute("c", c)
        g = pc.get_grid(ncols=2, nrows=2, x1=0, x2=10, y1=0, y2=10, method = "most_frequent", attr="c")
        self.assertTrue( (g.grid == np.array(((0, 5), (0, 6)))).all())
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
