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
import tempfile
import json
import ctypes
from thatsDEM2 import pointcloud

LOG = logging.getLogger(__name__)


class TestPointcloud(unittest.TestCase):

    def test_pointcloud_constructor1(self):
        LOG.info("Testing pointcloud constructor")
        pc = pointcloud.Pointcloud(np.ones((2, 2)), np.ones(
            2), some_attr=np.ones(2, dtype=np.uint8))
        self.assertIn("some_attr", pc.attributes)
        self.assertTrue(pc.some_attr.dtype == np.uint8)

    def test_pointcloud_constructor_bad(self):
        LOG.info("Testing pointcloud constructor -bad")
        with self.assertRaises(AssertionError):
            pc = pointcloud.Pointcloud(np.ones((2, 2)), np.ones(
                2), some_attr=np.ones(4, dtype=np.uint8))

    def test_pointcloud_empty_like(self):
        LOG.info("Testing pointcloud empty_like factory function")
        pc = pointcloud.Pointcloud(
            np.ones((2, 2)), np.ones(2), some_attr=np.ones(2))
        empty = pointcloud.empty_like(pc)
        self.assertSetEqual(pc.attributes, empty.attributes)
        self.assertEqual(empty.size, 0)

    def test_extend_pointcloud1(self):
        LOG.info("Testing pointcloud extension - bad")
        pc1 = pointcloud.Pointcloud(
            np.ones((2, 2)), np.ones(2), some_attr=np.ones(2))
        pc2 = pointcloud.Pointcloud(np.ones((2, 2)), np.ones(2))
        with self.assertRaises(ValueError):
            pc1.extend(pc2)

    def test_extend_pointcloud2(self):
        LOG.info("Testing pointcloud extension - ok")
        pc1 = pointcloud.Pointcloud(
            np.ones((2, 2)), np.ones(2), some_attr=np.ones(2))
        pc2 = pointcloud.Pointcloud(
            np.ones((2, 2)), np.ones(2), some_attr=np.ones(2) * 3)
        pc1.extend(pc2)
        self.assertEqual(pc1.size, 4)
        self.assertIn("some_attr", pc1.attributes)

    def test_extend_pointcloud3(self):
        LOG.info("Testing pointcloud extension - least common")
        pc1 = pointcloud.Pointcloud(np.ones((2, 2)), np.ones(
            2), some_attr=np.ones(2), some_other=np.ones(2))
        pc2 = pointcloud.Pointcloud(
            np.ones((2, 2)), np.ones(2), some_attr=np.ones(2) * 3)
        pc1.extend(pc2, least_common=True)
        self.assertSetEqual(pc1.attributes, {"some_attr"})

    def test_thin_pointcloud(self):
        LOG.info("Testing thin pointcloud")
        pc = pointcloud.Pointcloud(np.ones((5, 2)), np.ones(
            5), some_attr=np.ones(5), some_other=np.ones(5))
        M = np.array([1, 0, 1, 1, 0]).astype(np.bool)
        pc.thin(M)
        self.assertEqual(pc.size, M.sum())
        self.assertSetEqual(pc.attributes, {"some_attr", "some_other"})

    def test_cut_pointcloud(self):
        LOG.info("Testing cut poincloud")
        pc = pointcloud.Pointcloud(np.ones((5, 2)), np.ones(
            5), some_attr=np.ones(5), some_other=np.ones(5))
        M = np.array([1, 0, 1, 1, 0]).astype(np.bool)
        pc = pc.cut(M)
        self.assertEqual(pc.size, M.sum())
        self.assertSetEqual(pc.attributes, {"some_attr", "some_other"})

    def test_lidar_pointcloud(self):
        LOG.info("Testing lidar pointcloud")
        pc = pointcloud.LidarPointcloud(np.ones((3, 2)), np.ones(3), c=[
                                        2, 2, 3], some_attr=np.ones(3))
        self.assertItemsEqual(pc.get_classes(), [2, 3])
        self.assertEqual(pc.cut_to_class(2).size, 2)

    def test_lidar_pointcloud_chained_cut(self):
        LOG.info("Testing lidar pointcloud chained cut")
        # We want subclasses to return subclasses in cut
        pc = pointcloud.LidarPointcloud(np.ones((3, 2)), np.ones(3), c=[
                                        2, 2, 3], some_attr=np.ones(3))
        pc2 = pc.cut_to_class(3).cut_to_box(-10, -10, 10, 10).cut_to_class(3)
        self.assertEqual(pc2.size, 1)

    def test_sort_pointcloud(self):
        LOG.info("Test pointcloud sorting")
        r = np.linspace(0, np.pi * 2, 100)
        xy = np.column_stack((r * np.cos(r), r * np.sin(r))) * 5
        c = np.arange(xy.shape[0])
        pc = pointcloud.Pointcloud(xy, np.ones(xy.shape[0]), c=c)
        pc.sort_spatially(1, keep_sorting=True)
        self.assertTrue((c != pc.c).any())
        pc.sort_back()
        self.assertTrue((pc.c == c).all())

    def test_pointcloud_might_overlap(self):
        LOG.info("Test pointcloud sorting")
        pc1 = pointcloud.from_array(np.ones((10, 10)), [0, 1, 0, 10, 0, -1])
        pc2 = pointcloud.from_array(np.ones((10, 10)), [0, 1, 0, 5, 0, -1])
        self.assertTrue(pc1.might_overlap(pc2))
        pc1.affine_transformation_2d(T=(30, 30))
        self.assertFalse(pc1.might_overlap(pc2))

    def test_pointcloud_attributes(self):
        LOG.info("Test pointcloud attributes")
        pc = pointcloud.Pointcloud(
            np.ones((10, 2)), np.ones(10), a=np.arange(10))
        with self.assertRaises(pointcloud.InvalidArrayError):
            pc.xy = 10
        with self.assertRaises(pointcloud.InvalidArrayError):
            pc.a = "abc"
        # Should be ok
        pc.a = range(10, 0, -1)
        self.assertEqual(pc.a[0], 10)

    def test_pointcloud_min_filter(self):
        LOG.info("Test pointcloud min filter")
        pc = pointcloud.Pointcloud(
            ((0, 0), (1, 0), (2, 0), (3, 0)), (1, 2, 3, 4))
        pc.sort_spatially(2)
        z = pc.min_filter(1.5)
        self.assertTrue((z == (1, 1, 2, 3)).all())

    def _test_pointcloud_grid_filter(self, method, mean_val):
        LOG.info("Test pointcloud gridding, method: %s" % str(method))
        pc = pointcloud.from_array(
            np.arange(100).reshape((10, 10)), [0, 1, 0, 10, 0, -1])
        pc.sort_spatially(2)
        g = pc.get_grid(ncols=10, nrows=10, x1=0, x2=10, y1=0,
                        y2=10, attr="z", srad=2, method=method)
        self.assertEqual(g.shape, (10, 10))
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
        pc = pointcloud.from_array(
            np.arange(100).reshape((10, 10)), [0, 1, 0, 10, 0, -1])
        g = pc.get_grid(ncols=10, nrows=10, x1=0, x2=10, y1=0,
                        y2=10, srad=2, method="cellcount")
        self.assertTrue((g.grid == 1).all())

    def test_pointcloud_grid_density_filter(self):
        LOG.info("Test pointcloud gridding, method: density_filter")
        pc = pointcloud.from_array(
            np.arange(100).reshape((10, 10)), [0, 1, 0, 10, 0, -1])
        pc.sort_spatially(2)
        g = pc.get_grid(ncols=10, nrows=10, x1=0, x2=10, y1=0,
                        y2=10, srad=2, method="density_filter")
        self.assertGreater(g.grid.min(), 0.4)
        self.assertLess(g.grid.max(), 1.1)
        self.assertTrue((g.grid[:, 0] == g.grid[:, -1]).all())
        self.assertTrue((g.grid[0, :] == g.grid[-1, :]).all())

    def test_pointcloud_grid_by_function(self):
        LOG.info("Test pointcloud gridding, method: np.max")
        pc = pointcloud.from_array(
            np.arange(100).reshape((10, 10)), [0, 1, 0, 10, 0, -1])
        g = pc.get_grid(ncols=2, nrows=2, x1=0, x2=10,
                        y1=0, y2=10, method=np.max)
        self.assertTrue((g.grid == np.array(((44., 49.), (94., 99.)))).all())

    def test_pointcloud_grid_most_frequent(self):
        LOG.info("Test pointcloud gridding, method: most_frequent")
        pc = pointcloud.from_array(np.ones((10, 10)), [0, 1, 0, 10, 0, -1])
        c = (np.arange(100) % 10).astype(np.int32)
        pc.set_attribute("c", c)
        g = pc.get_grid(ncols=2, nrows=2, x1=0, x2=10, y1=0,
                        y2=10, method="most_frequent", attr="c")
        self.assertTrue((g.grid == np.array(((0, 5), (0, 6)))).all())

    def test_pointcloud_TIN(self):
        LOG.info("Test pointcloud gridding, method: triangulation")
        pc = pointcloud.from_array(np.ones((10, 10)), [0, 1, 0, 10, 0, -1])
        pc.triangulate()
        c = np.ones(100) * 5.0
        pc.set_attribute("c", c)
        g = pc.get_grid(ncols=10, nrows=10, x1=0, x2=10, y1=0,
                        y2=10, method="triangulation", attr="c")
        self.assertAlmostEqual(np.fabs(g.grid - 5.0).max(), 0)

    def test_ballcount_filter(self):
        LOG.info("Test pointcloud ballcount filter")
        pc = pointcloud.from_array(np.ones((10, 10)), [0, 1, 0, 10, 0, -1])
        pc.sort_spatially(2)
        n1 = pc.ballcount_filter(0.5)
        self.assertTrue((n1 == 1).all())
        n2 = pc.ballcount_filter(0.5, xy=pc.xy, z=pc.z + 2)
        self.assertTrue((n2 == 0).all())

    def test_nearest_filter(self):
        LOG.info("Test pointcloud nearest filter")
        pc = pointcloud.from_array(np.ones((10, 10)), [0, 1, 0, 10, 0, -1])
        pc.sort_spatially(2)
        idx = pc.nearest_filter(2, xy=pc.xy + 0.25)
        self.assertTrue((idx == np.arange(0, 100)).all())

    def test_spike_filter(self):
        LOG.info("Test pointcloud spike filter")
        pc = pointcloud.from_array(np.ones((2, 2)), [-0.5, 1, 0, 1.5, 0, -1])
        pc.extend(pointcloud.Pointcloud([[0.5, 0.5]], (-5.0,)))
        pc.sort_spatially(2)
        vlim = (0.4) ** 2
        M = pc.spike_filter(2, vlim)
        self.assertTrue(M.sum() == 1)
        self.assertEqual(pc.z[np.where(M)[0][0]], -5.0)

    @staticmethod
    def custom_filter(xy, z, slices, pc_xy, pc_z, frad2, nd_val, opt_params):
        m = 0.0
        n = 0
        for i in range(3):
            j1 = slices[i * 3]
            j2 = slices[i * 3 + 1]
            for j in range(j1, j2):
                m += pc_z[j]
                n += 1
        return m / n

    def test_custom_filter(self):
        LOG.info("Test pointcloud custom filter")
        pc = pointcloud.from_array(np.ones((10, 10)), [0, 1, 0, 10, 0, -1])
        c = np.ones(100) * 5.0
        pc.set_attribute("c", c)
        pc.sort_spatially(2)
        z = pc.apply_2d_filter(2, self.custom_filter, attr="c")
        self.assertTrue((z == 5).all())
    
    def test_3dmean_filter(self):
        LOG.info("Test pointcloud custom filter")
        pc = pointcloud.from_array(np.ones((10, 10)), [0, 1, 0, 10, 0, -1])
        c = np.ones(100) * 5.0
        pc.set_attribute("c", c)
        pc.sort_spatially(2)
        mc = pc.mean_3d_filter(1, attr="c")
        # LOG.info("mc is " + str(mc))
        self.assertTrue((mc == 5).all())

    def _get_named_temp_file(self, ext):
        f = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        f.close()
        os.unlink(f.name)  # now we have a unique name, nice :-)
        return f.name

    def test_dump_npz(self):
        LOG.info("Test pointcloud dump_npz /from_npz")
        pc = pointcloud.from_array(
            np.arange(100).reshape((10, 10)), [0, 1, 0, 10, 0, -1])
        pc.set_attribute("a", np.arange(100))
        path = self._get_named_temp_file(".npz")
        pc.dump_npz(path)
        pc_rest = pointcloud.from_npz(path)
        self.assertTrue((pc_rest.xy == pc_rest.xy).all())
        self.assertTrue((pc_rest.a == pc.a).all())
        os.remove(path)

    def test_dump_ogr(self):
        LOG.info("Test pointcloud dump_ogr")
        pc = pointcloud.from_array(
            np.arange(100).reshape((10, 10)), [0, 1, 0, 10, 0, -1])
        pc.set_attribute("a", np.arange(100))
        path = self._get_named_temp_file(".geojson")
        pc.dump_new_ogr_datasource(path, fmt="GEOJson")
        # TODO: restore again...
        with open(path) as f:
            obj = json.load(f)
            self.assertTrue("features" in obj)
            self.assertEqual(len(obj["features"]), pc.size)
        pc_restore = pointcloud.from_ogr(path)
        self.assertItemsEqual(pc.attributes, pc_restore.attributes)
        self.assertAlmostEqual(np.fabs(pc.xy - pc_restore.xy).max(), 0, 3)
        self.assertAlmostEqual(np.fabs(pc.z - pc_restore.z).max(), 0, 3)
        os.remove(path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
