# Original work Copyright (c) 2015, Danish Geodata Agency <gst@gst.dk>
# Modified work Copyright (c) 2015-2016, Geoboxers <info@geoboxers.com>
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
Methods to work easier with ogr geometries and numpy arrays in combination.
Contains some custom geospatial methods action on numpy arrays.
silyko, June 2016.
"""
import ctypes

import numpy as np
from osgeo import ogr

from thatsDEM2 import shared_libraries as sh

# Py2 to 3
try:
    basestring
except NameError:
    basestring = str
try:
    xrange
except NameError:
    xrange = range

XY_TYPE = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=['C', 'O', 'A', 'W'])
GRID_TYPE = np.ctypeslib.ndpointer(
    dtype=np.float64, ndim=2, flags=['C', 'O', 'A', 'W'])
Z_TYPE = np.ctypeslib.ndpointer(
    dtype=np.float64, ndim=1, flags=['C', 'O', 'A', 'W'])
MASK_TYPE = np.ctypeslib.ndpointer(
    dtype=np.bool, ndim=1, flags=['C', 'O', 'A', 'W'])
UINT32_TYPE = np.ctypeslib.ndpointer(
    dtype=np.uint32, ndim=1, flags=['C', 'O', 'A'])
HMAP_TYPE = np.ctypeslib.ndpointer(
    dtype=np.uint32, ndim=2, flags=['C', 'O', 'A'])
UINT8_VOXELS = np.ctypeslib.ndpointer(
    dtype=np.uint8, ndim=3, flags=['C', 'O', 'A', 'W'])
INT32_VOXELS = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=3, flags=['C', 'O', 'A', 'W'])
INT32_TYPE = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags=['C', 'O', 'A', 'W'])


# Load the library using np.ctypeslib
lib = np.ctypeslib.load_library(sh.LIB_FGEOM, sh.LIB_DIR)

##############
# corresponds to
# array_geometry.h
##############
# void p_in_buf(double *p_in, char *mout, double *verts, unsigned long np,
# unsigned long nv, double d)
lib.p_in_buf.argtypes = [XY_TYPE, MASK_TYPE, XY_TYPE,
                         ctypes.c_ulong, ctypes.c_ulong, ctypes.c_double]
lib.p_in_buf.restype = None
lib.p_in_poly.argtypes = [XY_TYPE, MASK_TYPE,
                          XY_TYPE, ctypes.c_uint, UINT32_TYPE, ctypes.c_uint]
lib.p_in_poly.restype = ctypes.c_int
lib.get_triangle_geometry.argtypes = [
    XY_TYPE, Z_TYPE, sh.LP_CINT, np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=[
            'C', 'O', 'A', 'W']), ctypes.c_int]
lib.get_triangle_geometry.restype = None
lib.get_normals.argtypes = [
    XY_TYPE, Z_TYPE, sh.LP_CINT, np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=2, flags=[
            'C', 'O', 'A', 'W']), ctypes.c_int]
lib.get_normals.restype = None
lib.mark_bd_vertices.argtypes = [
    MASK_TYPE, MASK_TYPE, sh.LP_CINT, MASK_TYPE, ctypes.c_int, ctypes.c_int]
lib.mark_bd_vertices.restype = None
# int fill_spatial_index(int *sorted_flat_indices, int *index, int
# npoints, int max_index)
lib.fill_spatial_index.argtypes = [
    INT32_TYPE, INT32_TYPE, ctypes.c_int, ctypes.c_int]
lib.fill_spatial_index.restype = ctypes.c_int


FILTER_FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_double,
                                    sh.LP_CDOUBLE,
                                    ctypes.c_double,
                                    sh.LP_CINT,
                                    sh.LP_CDOUBLE,
                                    sh.LP_CDOUBLE,
                                    ctypes.c_double,
                                    ctypes.c_double,
                                    ctypes.c_void_p)

lib.apply_filter.argtypes = (
    XY_TYPE,
    sh.LP_CDOUBLE,
    XY_TYPE,
    Z_TYPE,
    Z_TYPE,
    INT32_TYPE,
    Z_TYPE,
    ctypes.c_int,
    FILTER_FUNC_TYPE,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_void_p
)

lib.apply_filter.restype = None

# void pc_noise_filter(double *pc_xy, double *pc_z, double *z_out, double filter_rad,
#                      double zlim, double den_cut, int *spatial_index, double *header, int npoints);
# binning
# void moving_bins(double *z, int *nout, double rad, int n);
lib.moving_bins.argtypes = [Z_TYPE, INT32_TYPE, ctypes.c_double, ctypes.c_int]
lib.moving_bins.restype = None
# a triangle based filter
# void tri_filter_low(double *z, double *zout, int *tri, double cut_off,
# int ntri)
lib.tri_filter_low.argtypes = [Z_TYPE, Z_TYPE,
                               sh.LP_CINT, ctypes.c_double, ctypes.c_int]
lib.tri_filter_low.restype = None
# hmap filler
# void fill_it_up(unsigned char *out, unsigned int *hmap, int rows, int
# cols, int stacks);
lib.fill_it_up.argtypes = [UINT8_VOXELS, HMAP_TYPE] + [ctypes.c_int] * 3
lib.fill_it_up.restype = None
lib.find_floating_voxels.argtypes = [
    INT32_VOXELS, INT32_VOXELS] + [ctypes.c_int] * 4
lib.find_floating_voxels.restype = None

# unsigned long simplify_linestring(double *xy_in, double *xy_out, double dist_tol, unsigned long n_pts)
lib.simplify_linestring.argtypes = [XY_TYPE, XY_TYPE, ctypes.c_double, ctypes.c_ulong]
lib.simplify_linestring.restype = ctypes.c_ulong

# Names of defined filter functions
LIBRARY_FILTERS = ("mean_filter",
                   "median_filter",
                   "adaptive_gaussian_filter",
                   "min_filter",
                   "max_filter",
                   "var_filter",
                   "idw_filter",
                   "density_filter",
                   "distance_filter",
                   "nearest_filter",
                   "ballcount_filter",
                   "spike_filter",
                   "ray_mean_dist_filter",
                   "mean_3d_filter")


def apply_filter(along_xy, along_z,
                 pc_xy, pc_attr,
                 spatial_index,
                 index_header,
                 filter_func,
                 filter_rad,
                 nd_val,
                 params=None):
    """
    Apply a bultin library filter, or a filter defined by a python function.
    Args:
        along_xy: Numpy array of input points.
        along_z: Numpy array of z values if 3d-filter, else None.
        pc_xy: The points to apply the filter on.
        pc_attr: The values to apply the filter on (z if a geometric filter).
        spatial_index: Pointcloud spatial index (see Pointcloud.sort_spatially)
        index_header: Pointcloud index metadata header.
        filter_func: A name of one of the builtin filters, or a python callable.
        filter_rad: Filter radius (which the filter function will use as needed).
        nd_val: No data value.
        params: Optional addtional parameters. MUST be a ctypes.c_void_p pointer if not None.
    Returns:
        1d array of filtered values
    """
    out = np.zeros(along_xy.shape[0], dtype=np.float64)
    if callable(filter_func):
        func = FILTER_FUNC_TYPE(filter_func)
    else:
        if not isinstance(filter_func, basestring):
            raise ValueError("filter_func must be a name (string) or a callable.")
        if filter_func not in LIBRARY_FILTERS:
            raise ValueError("No builtin filter called " + filter_func)
        addr = ctypes.cast(getattr(lib, filter_func), ctypes.c_void_p).value
        func = FILTER_FUNC_TYPE(addr)
    if along_z is not None:
        # If using a 3d filter - construct pointer
        assert along_z.shape[0] == along_xy.shape[0]
        pz = along_z.ctypes.data_as(sh.LP_CDOUBLE)
    else:
        pz = None
    if params is not None and not isinstance(params, ctypes.c_void_p):
        raise ValueError("params must be None or a ctypes.c_void_p pointer!")
    lib.apply_filter(along_xy, pz, pc_xy, pc_attr, out, spatial_index,
                     index_header, along_xy.shape[0], func, filter_rad,
                     nd_val, params)
    return out


def binary_fill_gaps(M):
    """
    Fill small gaps between elements in a binary mask
    """
    N = np.zeros_like(M)
    lib.binary_fill_gaps(M, N, M.shape[0], M.shape[1])
    return N


def line_intersection(l1, l2):
    """
    Test whether two lines l1, l2 in 2d intersect.
    Args:
        l1: Numpy array of shape (2, 2)
        l2: Numpy array og shape (2, 2)
    Returns:
       Intersection point, lc1, lc2 (line coords) if lines are NOT colinear and intersect.
       If no intersection (or colinear) return None, None, None
    """
    v1 = l1[1] - l1[0]
    v2 = l2[0] - l2[1]
    v3 = l2[0] - l1[0]
    w = np.column_stack((v1, v2, v3))
    D2 = np.linalg.det(w[:, (0, 1)])
    if abs(D2) < 1e-10:
        return None, None, None  # TODO: fix here
    D1 = np.linalg.det(w[:, (0, 2)])
    D0 = np.linalg.det(w[:, (1, 2)])
    s1 = - D0 / D2
    s2 = D1 / D2
    if 0 <= s1 <= 1 and 0 <= s2 <= 1:
        return l1[0] + s1 * v1, s1, s2
    return None, None, None


def simplify_linestring(xy, dtol):
    """
    Simplify a 2D-linestring (xy)
    Args:
        xy: numpy array of shape (n,2) and dtype float64
        dtol: Distance tolerance.
    Returns:
        Simplified xy array.
    """
    if xy.shape[0] < 3:
        return xy
    xy_out = np.zeros_like(xy)
    n_out = lib.simplify_linestring(xy, xy_out, dtol, xy.shape[0])
    xy_out = xy_out[:n_out].copy()
    return xy_out


def moving_bins(z, rad):
    """
    Count points within a bin of size 2*rad around each point.
    Corresponds to a 'moving' histogram, or a 1d 'count filter'.
    """
    # Will sort input -- so no need to do that first...
    zs = np.sort(z).astype(np.float64)
    n_out = np.zeros(zs.shape, dtype=np.int32)
    lib.moving_bins(zs, n_out, rad, zs.shape[0])
    return zs, n_out


def tri_filter_low(z, tri, ntri, cut_off):
    """
    Triangulation based filtering of input z.
    Will test dz for each edge, and replace high point with low point if dz is larger than cut_off.
    Used to flatten steep triangles which connect e.g. a water point to a vegetation point on a tree
    """
    zout = np.copy(z)
    lib.tri_filter_low(z, zout, tri, cut_off, ntri)
    return zout


def ogrpoints2array(ogr_geoms):
    """
    Convert a list of OGR point geometries to a numpy array.
    Slow interface.
    """
    out = np.empty((len(ogr_geoms), 3), dtype=np.float64)
    for i in xrange(len(ogr_geoms)):
        out[i, :] = ogr_geoms[i].GetPoint()
    return out


def ogrmultipoint2array(ogr_geom, flatten=False):
    """
    Convert a OGR multipoint geometry to a numpy (2d or 3d) array.
    """
    t = ogr_geom.GetGeometryType()
    assert(t == ogr.wkbMultiPoint or t == ogr.wkbMultiPoint25D)
    ng = ogr_geom.GetGeometryCount()
    out = np.zeros((ng, 3), dtype=np.float64)
    for i in range(ng):
        out[i] = ogr_geom.GetGeometryRef(i).GetPoint()
    if flatten:
        out = out[:, 0:2].copy()
    return out


def ogrgeom2array(ogr_geom, flatten=True):
    """
    OGR geometry to numpy array dispatcher.
    Will just send the geometry to the appropriate converter based on geometry type.
    """
    t = ogr_geom.GetGeometryType()
    if t == ogr.wkbLineString or t == ogr.wkbLineString25D:
        return ogrline2array(ogr_geom, flatten)
    elif t == ogr.wkbPolygon or t == ogr.wkbPolygon25D:
        return ogrpoly2array(ogr_geom, flatten)
    elif t == ogr.wkbMultiPoint or t == ogr.wkbMultiPoint25D:
        return ogrmultipoint2array(ogr_geom, flatten)
    else:
        raise Exception("Unsupported geometry type: %s" %
                        ogr_geom.GetGeometryName())


def ogrpoly2array(ogr_poly, flatten=True):
    """
    Convert a OGR polygon geometry to a list of numpy arrays.
    The first element will be the outer ring. Subsequent elements correpsond to the boundary of holes.
    Will not handle 'holes in holes'.
    """
    ng = ogr_poly.GetGeometryCount()
    rings = []
    for i in range(ng):
        ring = ogr_poly.GetGeometryRef(i)
        arr = np.asarray(ring.GetPoints())
        if flatten and arr.shape[1] > 2:
            arr = arr[:, 0:2].copy()
        rings.append(arr)
    return rings


def ogrline2array(ogr_line, flatten=True):
    """
    Convert a OGR linestring geometry to a numpy array (of vertices).
    """
    t = ogr_line.GetGeometryType()
    assert(t == ogr.wkbLineString or t == ogr.wkbLineString25D)
    pts = ogr_line.GetPoints()
    # for an incompatible geometry ogr returns None... but does not raise a
    # python error...!
    if pts is None:
        if flatten:
            return np.empty((0, 2))
        else:
            return np.empty((0, 3))
    arr = np.asarray(pts)
    if flatten and arr.shape[1] > 2:
        arr = arr[:, 0:2].copy()
    return arr


def area_of_ring(xy):
    """Get the area of a closed ring"""
    # Should be a closed ring
    assert (xy[0] == xy[-1]).all()
    a = 0
    for i in range(xy.shape[0] - 1):
        a += np.linalg.det(xy[i:i + 2]) * 0.5
    return abs(a)


def area_of_polygon(rings):
    """Get the area of a polygon == list of numpy arrays"""
    a = area_of_ring(rings[0])
    if len(rings) > 1:
        for ring in rings[1:]:
            a -= area_of_ring(ring)
    return a


def points_in_buffer(points, vertices, dist):
    """
    Calculate a mask indicating whether points lie within a distance (given by dist) of a line,
    specified by the vertices arg.
    """
    out = np.empty((points.shape[0],), dtype=np.bool)  # its a byte, really
    lib.p_in_buf(points, out, vertices, points.shape[
                 0], vertices.shape[0], dist)
    return out


def get_triangle_geometry(xy, z, triangles, n_triangles):
    """
    Calculate the geometry of each triangle in a triangulation as an array with rows: (tanv2_i,bb_xy_i,bb_z_i).
    Here tanv2 is the squared tangent of the slope angle,
    bb_xy is the maximal edge of the planar bounding box, and bb_z_i the size of the vertical bounding box.
    Args:
        xy: The vertices of the triangulation.
        z: The z values of the vertices.
        triangles: ctypes pointer to a c-contiguous int array of triangles,
                   where each row contains the indices of the three vertices of a triangle.
        n_triangles: The number of triangles (rows in triangle array== size /3)
    Returns:
        Numpy array of shape (n,3) containing the geometry numbers for each triangle in the triangulation.
    """
    out = np.empty((n_triangles, 3), dtype=np.float32)
    lib.get_triangle_geometry(xy, z, triangles, out, n_triangles)
    return out


def get_normals(xy, z, triangles, n_triangles):
    """
    Compute normals vectors for a triangulation.
    Args:
        xy: The vertices of the triangulation.
        z: The z values of the vertices.
        triangles: ctypes pointer to a c-contiguous int32 array of triangles,
                   where each row contains the indices of the three vertices of a triangle.
        n_triangles: The number of triangles (rows in triangle array== size /3)
    Returns:
        Numpy array of shape (n,3) containing the geometry numbers for each triangle in the triangulation.
    """
    out = np.ones((n_triangles, 3), dtype=np.float64)
    lib.get_normals(xy, z, triangles, out, n_triangles)
    return out


def get_curvatures(xyz, triangles, inds=None):
    """
    Compute curvatures based on a (surface) triangulation.
    TODO: speedier implementation in c - this is a toy test.
    Args:
        xyz: Numpy array of vertices - shape (n, 3)
        triangles: Numpy integer array of triangles, shape(m, 3)
        inds: Indices for pts. in xyz to do - defaults to all.
    Return:
        curvatures: Numpy array of curvatures in specified pts,
        mean_slopes: Numpy array of mean (2d) slope in specified pts.

    """

    if inds is None:
        inds = np.arange(0, xyz.shape[0])
    curvatures = np.zeros(inds.shape[0], dtype=np.float64)
    m_slopes = np.zeros(inds.shape[0], dtype=np.float64)
    n = 0
    for i in inds:
        # iterate over first axis
        I, J = np.where(triangles == i)
        alpha = 0
        m_slope = 0
        n_edges = 0
        for j in range(I.size):
            trig = triangles[I[j]]
            p0 = xyz[trig[J[j]]]
            l1 = xyz[trig[(J[j] + 1) % 3]] - p0
            l2 = xyz[trig[(J[j] + 2) % 3]] - p0
            # slopes < 0 if cur pt is 'above' other
            # mean slope < 0 means, we lie above a 'mean plane...'
            # making z less, increases mean slope
            m_slope += l1[2] / np.sqrt(l1.dot(l1))
            m_slope += l2[2] / np.sqrt(l2.dot(l2))
            n_edges += 2
            d1 = np.sqrt(l1.dot(l1))
            d2 = np.sqrt(l2.dot(l2))
            alpha += np.arccos(np.dot(l1, l2) / (d1 * d2))
        curvatures[n] = 2 * np.pi - alpha
        if n_edges > 0:
            m_slopes[n] = m_slope / n_edges
        n += 1
    return curvatures, m_slopes


def get_bounds(geom):
    """Just return the bounding box for a geometry represented as a numpy array
    (or a list of arrays correpsponding to a polygon)."""
    if isinstance(geom, list):
        arr = geom[0]
    else:
        arr = geom
    bbox = np.empty((4,), dtype=np.float64)
    bbox[0:2] = np.min(arr[:, :2], axis=0)
    bbox[2:4] = np.max(arr[:, :2], axis=0)
    return bbox


def points2ogrpolygon(rings):
    """Construct a OGR polygon from an input point list (not closed)"""
    # input an iterable of 2d 'points', slow interface for large collections...
    poly = ogr.Geometry(ogr.wkbPolygon)
    if isinstance(rings, np.ndarray):  # just one 'ring'
        rings = [rings]

    for ring in rings:
        ogr_ring = ogr.Geometry(ogr.wkbLinearRing)
        for pt in ring:
            ogr_ring.AddPoint(pt[0], pt[1])
        ogr_ring.CloseRings()
        poly.AddGeometry(ogr_ring)
    return poly


def bbox_intersection(bbox1, bbox2):
    # simple intersection of two boxes given as (xmin,ymin,xmax,ymax)
    box = [-1, -1, -1, -1]
    box[0] = max(bbox1[0], bbox2[0])
    box[1] = max(bbox1[1], bbox2[1])
    box[2] = min(bbox1[2], bbox2[2])
    box[3] = min(bbox1[3], bbox2[3])
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return box


def bbox_to_polygon(bbox):
    """Convert a box given as (xmin,ymin,xmax,ymax) to a OGR polygon geometry."""
    points = ((bbox[0], bbox[1]), (bbox[2], bbox[1]),
              (bbox[2], bbox[3]), (bbox[0], bbox[3]))
    poly = points2ogrpolygon(points)
    return poly


def cut_geom_to_bbox(geom, bbox):
    # input a bounding box as returned from get_bounds...
    poly = bbox_to_polygon(bbox)
    return poly.Intersection(geom)


def points_in_polygon(points, rings):
    """
    Calculate a mask indicating whether points lie within a polygon.
    Args:
        points: 2d numpy array ( shape (n,2) ).
        rings: The list of rings (outer rings first) as returned by ogrpoly2array.
    Returns:
        1d numpy boolean array.
    """
    verts = np.empty((0, 2), dtype=np.float64)
    nv = []
    for ring in rings:
        if not (ring[-1] == ring[0]).all():
            raise ValueError("Polygon boundary not closed!")
        verts = np.vstack((verts, ring))
        nv.append(ring.shape[0])
    nv = np.asarray(nv, dtype=np.uint32)
    out = np.empty((points.shape[0],), dtype=np.bool)  # its a byte, really
    lib.p_in_poly(points, out, verts, points.shape[0], nv, len(rings))
    return out


def get_boundary_vertices(validity_mask, poly_mask, triangles):
    # Experimental: see pointcloud.py for explanation.
    out = np.empty_like(poly_mask)
    lib.mark_bd_vertices(
        validity_mask,
        poly_mask,
        triangles,
        out,
        validity_mask.shape[0],
        poly_mask.shape[0])
    return out


def linestring_displacements(xy):
    """
    Calculate the 'normal'/displacement vectors needed to buffer a line string (xy array of shape (n,2))
    """
    dxy = xy[1:] - xy[:-1]
    # should return a 1d array...
    ndxy = np.sqrt((dxy ** 2).sum(axis=1)).reshape((dxy.shape[0], 1))
    hat = np.column_stack((-dxy[:, 1], dxy[:, 0])) / ndxy  # dxy should be 2d
    normals = hat[0]
    # calculate the 'inner normals' - if any...
    if hat.shape[0] > 1:
        dots = (hat[:-1] * hat[1:]).sum(axis=1).reshape((hat.shape[0] - 1, 1))
        # dot of inner normal with corresponding hat should be = 1
        # <(v1+v2),v1>=1+<v1,v2>=<(v1+v2),v2>
        # assert ( not (dots==-1).any() ) - no 180 deg. turns!
        alpha = 1 / (1 + dots)
        # should be 2d - even with one row - else use np.atleast_2d
        inner_normals = (hat[:-1] + hat[1:]) * alpha
        normals = np.vstack((normals, inner_normals))
    normals = np.vstack((normals, hat[-1]))
    return normals


def project_onto_line(xy, x1, x2):
    """Line coords between 0 and 1 will be in between x2 and x1"""
    xy = np.atleast_2d(np.array(xy))
    r = x2 - x1
    n2 = r.dot(r)
    line_coords = (xy - x1).dot(r) / n2
    # return the internal 1d line coords and corresponding real xy coords
    return line_coords, line_coords.reshape((xy.shape[0], 1)) * r.reshape((1, 2)) + x1.reshape((1, 2))


def distance_to_polygon(xy, rings):
    """For each 2d-point in xy, calculate distance to polygon (list of Numpy arrays)."""
    raise NotImplementedError()  # Use c-func


def snap_to_polygon(xy_in, poly, dtol_v, dtol_bd=100):
    """
    Insert point(s) into a polygon [outer_ring, hole1, hole2...],
    where distance to bd. is minimal.
    Will 'snap' to an existing vertex if distance is less than dtol.
    Modifies polygon in-place if successful.
    Args:
        xy_in: The input pt.
        poly: 2D numpy arrays [outer_ring, hole1, ...] as returned from ogrpoly2array.
        dtol_v: The distance tolerance for snapping to a vertex.
        dtol_bd: The distance tolerance for snapping to bd.
    Returns:
        index_to_ring, index_in_ring_to_inserted_pt, has_snapped_to_vertex
        Will return -1, -1, False if distance tolerances are exceeded.
    """
    dmin = 1e10
    dtol_v2 = dtol_v ** 2
    dtol_bd2 = dtol_bd ** 2
    ring_min = -1
    imin = -1
    for ir, ring in enumerate(poly):
        d2 = ((ring - xy_in) ** 2).sum(axis=1)
        i = np.argmin(d2)
        if d2[i] < dtol_v2 and d2[i] < dmin:
            dmin = d2[i]
            ring_min = ir
            imin = i
    if ring_min >= 0:
        return ring_min, imin, True
    # OK - so no vertices found. Do it all again...
    # TODO: check for the lc_used in order to not insert a double pt.
    xy_insert = None
    for ir, ring in enumerate(poly):
        for i in range(ring.shape[0] - 1):
            p1 = ring[i]
            p2 = ring[i + 1]
            lc, xy = project_onto_line(xy_in, p1, p2)
            if 0 <= lc[0] <= 1:
                dist = (xy[0, 0] - xy_in[0]) ** 2 + (xy[0, 1] - xy_in[1]) ** 2
                if dist < dtol_bd2 and dist < dmin:
                    ring_min = ir
                    imin = i + 1
                    dmin = dist
                    xy_insert = xy[0]
    if ring_min >= 0:
        ring = poly[ring_min]
        ring = np.vstack((ring[:imin], xy_insert, ring[imin:]))
        poly[ring_min] = ring

    return ring_min, imin, False


def mesh_as_points(shape, geo_ref):
    """
    Construct a mesh of xy coordinates corresponding to the cell centers of a grid.
    Args:
        shape: (nrows,ncols)
        geo_ref: GDAL style georeference of grid.
    Returns:
        Numpy array of shape (nrows*ncols,2).
    """
    x = geo_ref[0] + geo_ref[1] * 0.5 + np.arange(0, shape[1]) * geo_ref[1]
    y = geo_ref[3] + geo_ref[5] * 0.5 + np.arange(0, shape[0]) * geo_ref[5]
    x, y = np.meshgrid(x, y)
    xy = np.column_stack((x.flatten(), y.flatten()))
    assert(xy.shape[0] == shape[0] * shape[1])
    return xy


def rasterize_polygon(poly_as_array, shape, geo_ref):
    """
    Return a boolean numpy mask with 1 for cells within polygon.
    Args:
       poly_as_array: A polygon as returned by ogrpoly2array (list of numpy arrays / rings)
       shape: Shape (nrows, ncols) of output array
       geo_ref: GDAL style georeference of grid.
    Returns:
        Numpy boolean 2d array.
    """
    xy = mesh_as_points(shape, geo_ref)
    return points_in_polygon(xy, poly_as_array).reshape(shape)


def unit_test(n=1000):
    verts = np.asarray(
        ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)), dtype=np.float64)
    pts = np.random.rand(n, 2).astype(np.float64)  # n points in unit square
    M = points_in_buffer(pts, verts, 2)
    assert M.sum() == n
    M = points_in_polygon(pts, [verts])
    assert M.sum() == n
    pts += (2.0, 2.0)
    M = points_in_polygon(pts, [verts])
    assert not M.any()


if __name__ == "__main__":
    unit_test()
