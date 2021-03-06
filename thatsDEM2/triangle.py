# Original work Copyright (c) 2015, Danish Geodata Agency <gst@gst.dk>
# Modified work Copyright (c) 2015-2016, Geoboxers <info@geoboxers.com>
#
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
"""
Triangulation classes.
Can use triangle by Jonathan Shewchuk if c extension is built against that.
Speedier implementation of finding simplices, compared to scipy.spatial.Delaunay.
silyko, June 2016.
"""

import ctypes
import numpy as np
from thatsDEM2.shared_libraries import *

# Load library directly via ctypes. Could also have used the numpy interface.
lib = ctypes.cdll.LoadLibrary(LIB_TRIPY)
# First see if compiled against triangle?
try:
    # Args and return types of c functions. Corresponds to a header file.
    lib.use_triangle.restype = LP_CINT
    lib.use_triangle.argtypes = [LP_CDOUBLE, ctypes.c_int, LP_CINT]
    # int *use_triangle_pslg(double *xy, int *segments, double *holes, int np, int nseg, int nholes, int *nt)
    lib.use_triangle_pslg.restype = LP_CINT
    lib.use_triangle_pslg.argtypes = [
        LP_CDOUBLE,
        LP_CINT,
        LP_CDOUBLE,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        LP_CINT]
    # void get_triangle_centers(double *xy, int *triangles, double *out, int n_trigs)
    lib.get_triangle_centers.restype = None
    lib.get_triangle_centers.argtypes = [LP_CDOUBLE, LP_CINT, LP_CDOUBLE, ctypes.c_int]
    lib.free_vertices.restype = None
    lib.free_vertices.argtypes = [LP_CINT]
    lib.get_triangles.argtypes = [LP_CINT, LP_CINT, LP_CINT, ctypes.c_int, ctypes.c_int]
    lib.get_triangles.restype = None
except AttributeError:
    # If triangle not available - use scipy.spatial
    from scipy.spatial import Delaunay
    HAS_TRIANGLE = False
else:
    HAS_TRIANGLE = True

# Declare additional functions
lib.free_index.restype = None
lib.free_index.argtypes = [ctypes.c_void_p]
lib.find_triangle.restype = None
lib.find_triangle.argtypes = [
    LP_CDOUBLE,
    LP_CINT,
    LP_CDOUBLE,
    LP_CINT,
    ctypes.c_void_p,
    LP_CCHAR,
    ctypes.c_int]

lib.inspect_index.restype = None
lib.inspect_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

lib.build_index.restype = ctypes.c_void_p
lib.build_index.argtypes = [LP_CDOUBLE, LP_CINT, ctypes.c_double, ctypes.c_int, ctypes.c_int]

lib.interpolate.argtypes = [
    LP_CDOUBLE,
    LP_CDOUBLE,
    LP_CDOUBLE,
    LP_CDOUBLE,
    ctypes.c_double,
    LP_CINT,
    ctypes.c_void_p,
    LP_CCHAR,
    ctypes.c_int]
lib.interpolate.restype = None
# void make_grid(double *base_pts,double *base_z, int *tri, float *grid,
# float tgrid, double nd_val, int ncols, int nrows, double cx, double cy,
# double xl, double yu, spatial_index *ind)
lib.make_grid.argtypes = [LP_CDOUBLE,
                          LP_CDOUBLE,
                          LP_CINT,
                          LP_CFLOAT,
                          LP_CFLOAT,
                          ctypes.c_float,
                          ctypes.c_int,
                          ctypes.c_int] + [ctypes.c_double] * 4 + [ctypes.c_void_p]
lib.make_grid.restype = None
# void make_grid_low(double *base_pts,double *base_z, int *tri, float
# *grid,  float nd_val, int ncols, int nrows, double cx, double cy, double
# xl, double yu, double cut_off, spatial_index *ind)
lib.make_grid_low.argtypes = [LP_CDOUBLE,
                              LP_CDOUBLE,
                              LP_CINT,
                              LP_CFLOAT,
                              ctypes.c_float,
                              ctypes.c_int,
                              ctypes.c_int] + [ctypes.c_double] * 5 + [ctypes.c_void_p]
lib.make_grid_low.restype = None

lib.optimize_index.argtypes = [ctypes.c_void_p]
lib.optimize_index.restype = None


class TriangulationBase(object):
    """Triangulation class inspired by scipy.spatial.Delaunay
    Uses Triangle to do the hard work. Automatically builds an index.
    """
    vertices = None  # actually a pointer to int array of triangles
    index = None
    points = None
    segments = None
    holes = None
    ntrig = None

    def __del__(self):
        """Destructor"""
        if self.vertices is not None:
            lib.free_vertices(self.vertices)
            self.vertices = None
        if self.index is not None:
            lib.free_index(self.index)
            self.index = None

    def validate_points(self, points, ndim=2, dtype=np.float64):
        # ALL this stuff is not needed if we use numpys ctypeslib interface - TODO.
        if not isinstance(points, np.ndarray):
            raise ValueError("Input points must be a Numpy ndarray")
        ok = points.flags["ALIGNED"] and points.flags[
            "C_CONTIGUOUS"] and points.flags["OWNDATA"] and points.dtype == dtype
        if (not ok):
            raise ValueError(
                "Input points must have flags 'ALIGNED','C_CONTIGUOUS','OWNDATA' and data type %s" %
                dtype)
        # TODO: figure out something useful here....
        if points.ndim != ndim or (ndim == 2 and points.shape[1] != 2):
            raise ValueError("Bad shape of input - points:(n,2) z: (n,), indices: (n,)")

    def interpolate(self, z_base, xy_in, nd_val=-999, mask=None):
        """
        Barycentric interpolation of input points xy_in based on values z_base in vertices.
        Points outside triangulation gets nd_val
        """
        self.validate_points(xy_in)
        self.validate_points(z_base, 1)
        if z_base.shape[0] != self.points.shape[0]:
            raise ValueError(
                "There must be exactly the same number of input zs as the number of triangulated points.")
        if mask is not None:
            if mask.shape[0] != self.ntrig:
                raise ValueError("Validity mask size differs from number of triangles")
            self.validate_points(mask, ndim=1, dtype=np.bool)
            pmask = mask.ctypes.data_as(LP_CCHAR)
        else:
            pmask = None
        out = np.empty((xy_in.shape[0],), dtype=np.float64)
        lib.interpolate(xy_in.ctypes.data_as(LP_CDOUBLE), self.points.ctypes.data_as(LP_CDOUBLE),
                        z_base.ctypes.data_as(LP_CDOUBLE),
                        out.ctypes.data_as(LP_CDOUBLE), nd_val, self.vertices, self.index, pmask, xy_in.shape[0])
        return out

    def make_grid(self, z_base, ncols, nrows, xl, cx, yu, cy, nd_val=-999, return_triangles=False):
        """
        Interpolate a grid using (barycentric) interpolation.
        Args:
            z_base: The values to interpolate (numpy 1d array, float64).
            ncols: number of columns.
            nrows: number of rows.
            xl: Left edge / corner (GDAL style).
            cx: Horisontal cell size.
            yu: Upper edge / corner (GDAL style).
            cy: Vertical cell size (positive).
            nd_val: output no data value.
            return_triangles: bool, if True also return a grid containing triangle bounding box sizes.
        Returns:
            Numpy 2d-arrray (float64) (and numpy 2d float32 array if return_triangles=True)
        """
        # void make_grid(double *base_pts,double *base_z, int *tri, double *grid,
        # double nd_val, int ncols, int nrows, double cx, double cy, double xl,
        # double yu, spatial_index *ind)
        if z_base.shape[0] != self.points.shape[0]:
            raise ValueError(
                "There must be exactly the same number of input zs as the number of triangulated points.")
        grid = np.empty((nrows, ncols), dtype=np.float32)
        if return_triangles:
            t_grid = np.zeros((nrows, ncols), dtype=np.float32)
            p_t_grid = t_grid.ctypes.data_as(LP_CFLOAT)
        else:
            p_t_grid = None
        lib.make_grid(
            self.points.ctypes.data_as(LP_CDOUBLE),
            z_base.ctypes.data_as(LP_CDOUBLE),
            self.vertices,
            grid.ctypes.data_as(LP_CFLOAT),
            p_t_grid,
            nd_val,
            ncols,
            nrows,
            cx,
            cy,
            xl,
            yu,
            self.index)
        if return_triangles:
            return grid, t_grid
        else:
            return grid

    def make_grid_low(self, z_base, ncols, nrows, xl, cx, yu, cy, nd_val=-999, cut_off=1.5):
        """Experimental: gridding avoiding steep edges"""
        # void make_grid(double *base_pts,double *base_z, int *tri, double *grid,
        # double nd_val, int ncols, int nrows, double cx, double cy, double xl,
        # double yu, spatial_index *ind)
        if z_base.shape[0] != self.points.shape[0]:
            raise ValueError(
                "There must be exactly the same number of input zs as the number of triangulated points.")
        grid = np.empty((nrows, ncols), dtype=np.float32)
        lib.make_grid_low(
            self.points.ctypes.data_as(LP_CDOUBLE),
            z_base.ctypes.data_as(LP_CDOUBLE),
            self.vertices,
            grid.ctypes.data_as(LP_CFLOAT),
            nd_val,
            ncols,
            nrows,
            cx,
            cy,
            xl,
            yu,
            cut_off,
            self.index)
        return grid

    def get_triangles(self, indices=None):
        """Copy allocated triangles to numpy (n,3) int32 array. Invalid indices give (-1,-1,-1) rows."""
        if indices is None:
            indices = np.arange(0, self.ntrig).astype(np.int32)
        self.validate_points(indices, 1, np.int32)
        out = np.empty((indices.shape[0], 3), dtype=np.int32)
        lib.get_triangles(
            self.vertices,
            indices.ctypes.data_as(LP_CINT),
            out.ctypes.data_as(LP_CINT),
            indices.shape[0],
            self.ntrig)
        return out

    def get_triangle_centers(self):
        """
        Calculate triangle center of masses.
        Returns:
            Numpy 2d array of shape (ntriangles,2)
        """
        out = np.empty((self.ntrig, 2), dtype=np.float64)
        lib.get_triangle_centers(
            self.points.ctypes.data_as(LP_CDOUBLE),
            self.vertices,
            out.ctypes.data_as(LP_CDOUBLE),
            self.ntrig)
        return out

    def rebuild_index(self, cs):
        """Rebuild index with another cell size"""
        lib.free_index(self.index)
        self.index = lib.build_index(
            self.points.ctypes.data_as(LP_CDOUBLE),
            self.vertices,
            cs,
            self.points.shape[0],
            self.ntrig)

    def optimize_index(self):
        """
        Only shrinks index slightly in memory.
        """
        # TODO: Should also sort index after areas of intersections between cells and triangles.
        lib.optimize_index(self.index)

    def inspect_index(self):
        """Return info as text"""
        info = ctypes.create_string_buffer(1024)
        lib.inspect_index(self.index, info, 1024)
        return info.value

    def find_triangles(self, xy, mask=None):
        """
        Finds triangle indices of input points. Returns -1 if no triangles is found.
        Can be used to implement a point in polygon algorithm (for convex polygons without holes).
        Args:
            xy: The points in which to look for containing triangles.
            mask: optional, A 1d validity mask marking validity of triangles.
        Returns:
            Numpy 1d int32 array containing triangles indices. -1 is used to indicate no (valid) triangle.
        """
        self.validate_points(xy)
        out = np.empty((xy.shape[0],), dtype=np.int32)
        if mask is not None:
            if mask.shape[0] != self.ntrig:
                raise ValueError("Validity mask size differs from number of triangles")
            self.validate_points(mask, ndim=1, dtype=np.bool)
            pmask = mask.ctypes.data_as(LP_CCHAR)
        else:
            pmask = None
        lib.find_triangle(
            xy.ctypes.data_as(LP_CDOUBLE),
            out.ctypes.data_as(LP_CINT),
            self.points.ctypes.data_as(LP_CDOUBLE),
            self.vertices,
            self.index,
            pmask,
            xy.shape[0])
        return out


class ShewchukTriangulation(TriangulationBase):
    """
    TriangulationBase implementation. Will construct a triangulation and an index of triangles.
    Requires triangle by Jonathan Richard Shewchuk.
    """

    def __init__(self, points, cs=-1):
        if not HAS_TRIANGLE:
            raise Exception("Requires that libtripy is built against triangle!")
        self.validate_points(points)
        self.points = points
        nt = ctypes.c_int(0)
        self.vertices = lib.use_triangle(
            points.ctypes.data_as(LP_CDOUBLE),
            points.shape[0],
            ctypes.byref(nt))
        self.ntrig = nt.value
        self.index = lib.build_index(
            points.ctypes.data_as(LP_CDOUBLE),
            self.vertices,
            cs,
            points.shape[0],
            self.ntrig)
        if self.index is None:
            raise Exception("Failed to build index...")


class QhullTriangulation(TriangulationBase):
    """
    Triangulation class using scipy.spatial.Delaunay instead of triangle.
    Slower and not as robust as triangle.
    """

    def __init__(self, points, cs=-1):
        self.validate_points(points)
        # For precision reasons we seem to need to transform to center of mass
        self.cm = points.mean(axis=0)
        self.points = points - self.cm
        self.delaunay = Delaunay(self.points)
        # for old scipy version:
        if hasattr(self.delaunay, "simplices"):
            self.tri_array = self.delaunay.simplices
        else:
            self.tri_array = self.delaunay.vertices
        self.vertices = self.tri_array.ctypes.data_as(LP_CINT)
        self.ntrig = self.tri_array.shape[0]
        self.index = lib.build_index(
            self.points.ctypes.data_as(LP_CDOUBLE),
            self.vertices,
            cs,
            points.shape[0],
            self.ntrig)
        if self.index is None:
            raise Exception("Failed to build index...")

    def __del__(self):
        """Destructor - override default, don't free triangle array"""
        if self.index is not None:
            lib.free_index(self.index)
            self.index = None

    def find_triangles_scipy(self, xy):
        return self.delaunay.find_simplex(xy - self.cm)

    def get_triangles(self):
        return self.delaunay.tri_array

    def get_triangle_centers(self):
        T = self.tri_array
        p = self.points[T[:, 0]]
        p += self.points[T[:, 1]]
        p += self.points[T[:, 2]]
        return p / 3 + self.cm

    def make_grid(self, z_base, ncols, nrows, xl, cx, yu, cy, nd_val=-999, return_triangles=False):
        # We need to shift towards center of mass here first.
        return super(QhullTriangulation, self).make_grid(z_base, ncols, nrows, xl - self.cm[0],
                                                         cx, yu - self.cm[1], cy, nd_val, return_triangles)

    def make_grid_low(self, *args, **kwargs):
        raise NotImplementedError()

    def find_triangles(self, xy, mask=None):
        # We need to shift towards center of mass here first.
        return super(QhullTriangulation, self).find_triangles(xy - self.cm, mask)

    def interpolate(self, z_base, xy_in, nd_val=-999, mask=None):
        return super(QhullTriangulation, self).interpolate(z_base, xy_in - self.cm, nd_val, mask)


if HAS_TRIANGLE:
    Triangulation = ShewchukTriangulation
else:
    Triangulation = QhullTriangulation


def using_triangle():
    """Just a convenience for a user to check whether libtripy was built against triangle."""
    return HAS_TRIANGLE


class PolygonData(object):
    """Helper class for handling data of a PSLG-triangulation"""

    def __init__(self, rings, **attrs):
        """
        Args:
            rings: list of numpy arrays (outer_ring, inner_ring1,...)
            attrs: Attributes to store for each vertex (e.g. z)
                   Eeach item must be a list of arrays of same shape as rings.
        """
        # first element in rings is outer ring, rest is holes - GEOS rings tend
        # to be closed,  we dont want that.
        # attr is an attribute of vertices - like z.
        self._points = np.empty((0, 2), dtype=np.float64)
        self._ring_slices = []  # start index, stop_index in self.points for each ring
        self._segments = np.empty((0, 2), dtype=np.int32)
        self._holes = np.zeros((0, 2), dtype=np.float64)
        self._inner_points = np.empty((0, 2), dtype=np.float64)  # points not 'inner' segments
        self._attrs = {a: np.empty(0, dtype=np.float64) for a in attrs}
        self._inner_attrs = {a: np.empty(0, dtype=np.float64) for a in attrs}
        for i, ring in enumerate(rings):
            _attrs = {a: attrs[a][i] for a in attrs}
            self._add_ring(rings[i], is_hole=(i > 0), **_attrs)

    def _add_ring(self, ring, is_hole=False, **attrs):
        """Add some more holes / rings"""
        if (ring[0, :] == ring[-1, :]).all():
            # unclose
            ring = ring[:-1]
            for a in attrs:
                attrs[a] = attrs[a][:-1]
        if is_hole:
            # add a hole center - assuming hole is convex!
            self._holes = np.vstack((self._holes, ring.mean(axis=0)))
        segments = np.zeros((ring.shape[0], 2), dtype=np.int32)
        segments[:, 0] = np.arange(self._points.shape[0], self._points.shape[0] + ring.shape[0])
        segments[:-1, 1] = np.arange(self._points.shape[0] + 1, self._points.shape[0] + ring.shape[0])
        segments[-1, 1] = self._points.shape[0]
        self._ring_slices.append((self._points.shape[0], self._points.shape[0] + ring.shape[0]))
        self._points = np.vstack((self._points, ring))
        for a in attrs:
            self._attrs[a] = np.concatenate((self._attrs[a], attrs[a]))
        self._segments = np.vstack((self._segments, segments))

    def add_holes(self, rings, **attrs):
        # Add some more holes
        if not set(attrs.keys()).issubset(set(self._attrs.keys())):
            raise ValueError("Cannot create new attrs here.")
        for i, ring in enumerate(rings):
            _attrs = {a: attrs[a][i] for a in attrs}
            self._add_ring(ring, is_hole=True, **_attrs)

    def add_points_and_segments(self, xy, segments, **attrs):
        """Add some more points and segments."""
        if not set(attrs.keys()).issubset(set(self._attrs.keys())):
            raise ValueError("Cannot create new attrs here.")

        n = self._points.shape[0]
        self._points = np.vstack((self._points, xy))
        assert segments.max() < xy.shape[0]
        assert segments.min() >= 0
        self._segments = np.vstack((self._segments, segments + n))
        for a in attrs:
            self._attrs[a] = np.concatenate((self._attrs[a], attrs[a]))

    def add_inner_points(self, xy, **attrs):
        """Add some points WITHOUT segments.
        We might want to edit these later on, so handy too keep those seperate.
        """
        if not set(attrs.keys()).issubset(set(self._attrs.keys())):
            raise ValueError("Cannot create new attrs here.")
        self._inner_points = np.vstack((self._inner_points, xy))
        for a in attrs:
            self._inner_attrs[a] = np.concatenate((self._inner_attrs[a], attrs[a]))

    @property
    def inner_point_indices(self):
        # Return indices of inner_points in self.points
        return np.arange(self._points.shape[0], self._points.shape[0] + self._inner_points.shape[0])

    @property
    def n_inner_points(self):
        return self._inner_points.shape[0]

    @property
    def points(self):
        return np.vstack((self._points, self._inner_points))

    @property
    def segments(self):
        return self._segments

    @property
    def holes(self):
        return self._holes

    def get_ring(self, i):
        # Return a view of the i'th ring
        if i >= len(self._ring_slices):
            raise ValueError("Too large index, not that many rings.")
        i0, i1 = self._ring_slices[i]
        return self._points[i0: i1]

    def get_rings(self):
        return [self._points[i0: i1] for i0, i1 in self._ring_slices]

    def attribute(self, a):
        # return a stacked attributte
        return np.concatenate((self._attrs[a], self._inner_attrs[a]))

    def thin_inner_points(self, M):
        # Mask M must be relative to self._inner_pts
        self._inner_points = self._inner_points[M]
        for a in self._inner_attrs:
            if self._inner_attrs[a].shape[0] > 0:
                self._inner_attrs[a] = self._inner_attrs[a][M]


class PSLGTriangulation(TriangulationBase):
    """
    TriangulationBase implementation. Triangulate a polygon / graph.
    Requires triangle by Jonathan Richard Shewchuk.
    """

    def __init__(self, points, segments, hole_points=None, cs=-1):  # low cs will speed up point in polygon
        if not HAS_TRIANGLE:
            raise Exception("Requires that libtripy is built against triangle!")
        self.segments = np.require(segments, dtype=np.int32, requirements=['A', 'O', 'C'])
        self.points = np.require(points, dtype=np.float64, requirements=['A', 'O', 'C'])
        if (hole_points is not None) and hole_points.shape[0] > 0:
            self.holes = np.require(hole_points, dtype=np.float64, requirements=['A', 'O', 'C'])
            assert self.holes.ndim == 2
            p_holes = self.holes.ctypes.data_as(LP_CDOUBLE)
            n_holes = self.holes.shape[0]
        else:
            self.holes = None
            p_holes = None
            n_holes = 0
        assert self.points.ndim == 2
        assert self.segments.ndim == 2
        assert self.segments.max() < self.points.shape[0]
        assert self.segments.min() >= 0
        assert points.shape[0] >= 3
        nt = ctypes.c_int(0)
        # int *use_triangle_pslg(double *xy, int *segments, double *holes, int np, int nseg, int nholes, int *nt)
        self.vertices = lib.use_triangle_pslg(
            self.points.ctypes.data_as(LP_CDOUBLE),
            self.segments.ctypes.data_as(LP_CINT),
            p_holes,
            self.points.shape[0],
            self.segments.shape[0],
            n_holes,
            ctypes.byref(nt))
        self.ntrig = nt.value
        self.index = lib.build_index(
            self.points.ctypes.data_as(LP_CDOUBLE),
            self.vertices,
            cs,
            self.points.shape[0],
            self.ntrig)
        if self.index is None:
            raise Exception("Failed to build index...")

    def inner_points(self):
        # Return a mask indicating whether a point is not on a segment
        M = np.ones(self.points.shape[0], dtype=np.bool)
        M[self.segments[:, 0]] = 0
        M[self.segments[:, 1]] = 0
        return M

    def points_in_polygon(self, xy):
        # based on what the index cell size is, this can be really fast and very robust!
        I = self.find_triangles(xy)
        return (I >= 0)
