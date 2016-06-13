# Copyright (c) 2016, Geoboxers <info@geoboxers.com>
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

"""
A really, really small wrapper around osgeo.osr making the python interface nicer.
silyko, June 2016
"""
from osgeo import osr
import numpy as np


def from_epsg(code):
    """Wrapper around what could have been a classmethod in osr.SpatialReference"""
    srs = osr.SpatialReference()
    ok = srs.ImportFromEPSG(code)
    if ok != 0:
        raise ValueError("Unable to import from EPSG: %d" % code)
    return srs


def from_proj4(proj4_def):
    """Wrapper around what could have been a classmethod in osr.SpatialReference"""
    srs = osr.SpatialReference()
    ok = srs.ImportFromProj4(proj4_def)
    if ok != 0:
        raise ValueError("Unable to import from PROJ4: %s" % proj4_def)
    return srs


def from_string(some_str):
    """Utility, where we can specify e.g. EPSG:25832, proj4 string, etc..."""
    some_str = some_str.strip()
    if some_str.lower().startswith("epsg:"):
        code = int(some_str.lower()[5:])
        return from_epsg(code)
    if "+proj" in some_str:
        return from_proj4(some_str)
    # else wkt?
    srs = osr.SpatialReference(some_str)
    return srs


def transform_array(transform, pts_in):
    """
    Transform a numpy array, and make sure to return a numpy array also.
    Args:
        transform: osr.CoordinateTransformation
        pts_in: numpy array of shape (n, 2) or (n, 3) or a list / tuple of points.
    Returns:
        numpy (float64) array of shape (n, 2) or (n, 3)
    """
    if isinstance(pts_in, np.ndarray):
        dim = pts_in.shape[1]
    else:
        dim = len(pts_in[0])
    out = np.asarray(transform.TransformPoints(pts_in), dtype=np.float64)
    if dim == 2:
        return out[:, :2].copy()
    return out
