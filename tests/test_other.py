# -*- coding: utf-8 -*-
"""
Unittests for various thatsDEM2 modules
@author: simlk
"""
import unittest
import logging
import numpy as np
from osgeo import ogr
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
