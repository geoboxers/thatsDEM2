# -*- coding: utf-8 -*-
"""
Unittests for various thatsDEM2 modules
@author: simlk
"""
import unittest
import logging
import numpy as np
from osgeo import ogr, osr, gdal
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
        M = np.zeros((6, 6), dtype=np.bool)
        M[0, :] = 1
        M[:3, 0] = 1
        M[:, 5] = 1
        M[5, :] = 1
        path = grid.walk_mask(M, (0, 0), (5, 0))
        self.assertEqual(path.shape[0], 14)

    def test_polygon_area(self):
        LOG.info("Testing polygon area.")
        poly = ogr.Geometry(ogr.wkbPolygon)
        ring1 = ogr.Geometry(ogr.wkbLinearRing)
        ring2 = ogr.Geometry(ogr.wkbLinearRing)
        for pt in ((0, 0), (5, 0), (5, 5), (0, 5), (0, 0)):
            ring1.AddPoint(*pt)
        for pt in ((1, 1), (3, 1), (3, 3), (1, 1)):
            ring2.AddPoint(*pt)
        poly.AddGeometry(ring1)
        poly.AddGeometry(ring2)
        poly_as_array = array_geometry.ogrpoly2array(poly)
        self.assertEqual(len(poly_as_array), 2)
        self.assertAlmostEqual(poly.GetArea(), array_geometry.area_of_polygon(poly_as_array))

    def test_grid_warp(self):
        LOG.info("Testing Grid.warp")
        arr = np.random.rand(100, 100) * 100
        # 10 by 10 degrees input extent
        geo_ref = [-5, 0.1 , 0, 5, 0, -0.1]
        srs_in = osr.SpatialReference()
        srs_in.ImportFromEPSG(4326)
        srs_out = osr.SpatialReference()
        # approximate output cellsize 11 km
        srs_out.ImportFromEPSG(3857)
        g_in = grid.Grid(arr, geo_ref, nd_val=-999, srs=srs_in.ExportToWkt())
        g_out = g_in.warp(srs_out.ExportToWkt())
        self.assertEqual(g_out.shape[0], 101)
        self.assertLessEqual(g_out.grid.max(), arr.max())
        self.assertGreaterEqual(g_out.grid[g_out.grid != -999].min(), arr.min())
        g_back = g_out.warp(srs_in.ExportToWkt(), cx=0.1, cy=0.1, out_extent=g_in.extent)
        self.assertEqual(g_back.shape[0], 100)
        self.assertEqual(g_back.shape[1], 100)
        self.assertAlmostEqual(np.fabs(g_back.geo_ref - geo_ref).max(), 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
