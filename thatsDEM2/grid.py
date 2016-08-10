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
#
"""
Contains a grid abstraction class and some useful grid construction methods.
Facilitates working with numpy arrays and GDAL datasources in combination.
silyko, June 2016.
"""
import numpy as np
import os
from osgeo import gdal, osr
import ctypes
import logging
try:
    import scipy.ndimage as image
except:
    HAS_NDIMAGE = False
else:
    HAS_NDIMAGE = True
from thatsDEM2.shared_libraries import *
LOG = logging.getLogger(__name__)
XY_TYPE = np.ctypeslib.ndpointer(dtype=np.float64, flags=['C', 'O', 'A', 'W'])
GRID_TYPE = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=['C', 'O', 'A', 'W'])
GRID32_TYPE = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags=['C', 'O', 'A', 'W'])
MASK2D_TYPE = np.ctypeslib.ndpointer(dtype=np.bool, ndim=2, flags=['C', 'O', 'A', 'W'])
Z_TYPE = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=['C', 'O', 'A', 'W'])
UINT32_TYPE = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags=['C', 'O', 'A', 'W'])
INT32_GRID_TYPE = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags=['C', 'O', 'A', 'W'])
INT32_TYPE = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags=['C', 'O', 'A', 'W'])
GEO_REF_ARRAY = ctypes.c_double * 4
# Load the library
lib = np.ctypeslib.load_library(LIB_GRID, LIB_DIR)
# void wrap_bilin(double *grid, double *xy, double *out, double *geo_ref,
# double nd_val, int nrows, int ncols, int npoints)

lib.wrap_bilin.argtypes = [
    GRID_TYPE,
    XY_TYPE,
    Z_TYPE,
    LP_CDOUBLE,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int]
lib.wrap_bilin.restype = None
# DLL_EXPORT void resample_grid(double *grid, double *out, double
# *geo_ref, double *geo_ref_out, double nd_val, int nrows, int ncols, int
# nrows_out, int ncols_out)
lib.resample_grid.argtypes = [
    GRID_TYPE,
    GRID_TYPE,
    LP_CDOUBLE,
    LP_CDOUBLE,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int]
lib.resample_grid.restype = None
# void grid_most_frequent_value(int *sorted_indices, int *values, int
# *out, int vmin,int vmax,int nd_val, int n)
lib.grid_most_frequent_value.argtypes = [
    INT32_TYPE,
    INT32_TYPE,
    INT32_GRID_TYPE,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int]
lib.grid_most_frequent_value.restype = None


# int flood_cells(float *dem, float cut_off, char *mask, char *mask_out,
# int nrows, int ncols)
lib.flood_cells.argtypes = [GRID32_TYPE, ctypes.c_float,
                            MASK2D_TYPE, MASK2D_TYPE] + [ctypes.c_int] * 2
lib.flood_cells.restype = ctypes.c_int
# void masked_mean_filter(float *dem, float *out, char *mask, int
# filter_rad, int nrows, int ncols)
lib.masked_mean_filter.argtypes = [
    GRID32_TYPE, GRID32_TYPE, MASK2D_TYPE] + [ctypes.c_int] * 3

lib.masked_mean_filter.restype = None

lib.binary_fill_gaps.argtypes = [MASK2D_TYPE,
                                 MASK2D_TYPE, ctypes.c_int, ctypes.c_int]
lib.binary_fill_gaps.restype = None

# unsigned long walk_mask(char *M, int *start, int *end, int *path, unsigned long buf_size, int nrows, int ncols);
lib.walk_mask.argtypes = [MASK2D_TYPE, INT32_TYPE, INT32_TYPE, INT32_GRID_TYPE, ctypes.c_ulong] + [ctypes.c_int] * 2
lib.walk_mask.restype = ctypes.c_ulong

# COMPRESSION OPTIONS FOR SAVING GRIDS AS GTIFF
DCO = ["TILED=YES", "COMPRESS=LZW"]

# Kernels for hillshading
ZT_KERNEL = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float32)  # Zevenberg-Thorne
H_KERNEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # Horn


def from_gdal(path, upcast=False):
    """
    Open a 1-band grid from a GDAL datasource.
    Args:
        path: GDAL connection string
        upcast: bool, indicates whether to upcast dtype to float64
    Returns:
        grid.Grid object
    """
    ds = gdal.Open(path)
    a = ds.ReadAsArray()
    if upcast:
        a = a.astype(np.float64)
    geo_ref = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    srs = osr.SpatialRefernece(wkt) if wkt else None
    nd_val = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    return Grid(a, geo_ref, nd_val, srs=srs)


def bilinear_interpolation(grid, xy, nd_val, geo_ref=None):
    """
    Perform bilinear interpolation in a grid. Will call a c-library extension.
    Args:
        grid: numpy array (of type numpy.float64)
        xy: numpy array of shape (n,2) (and dtype numpy.float64). The points to interpolate values for.
        nd_val: float, output no data value.
        geo_ref: iterable of floats: (xulcenter, hor_cellsize, yulcenter, vert_cellsize).
                 NOT GDAL style georeference. If None xy is assumed to be in array coordinates.
    Returns:
        A 1d, float64 numpy array containing the interpolated values.
    """

    if geo_ref is not None:
        if len(geo_ref) != 4:
            raise Exception("Geo reference should be sequence of len 4, xulcenter, cx, yulcenter, cy")
        geo_ref = GEO_REF_ARRAY(*geo_ref)
    p_geo_ref = ctypes.cast(geo_ref, LP_CDOUBLE)  # null or pointer to geo_ref
    grid = np.require(grid, dtype=np.float64, requirements=['A', 'O', 'C', 'W'])
    xy = np.require(xy, dtype=np.float64, requirements=['A', 'O', 'C', 'W'])
    out = np.zeros((xy.shape[0],), dtype=np.float64)
    lib.wrap_bilin(grid, xy, out, p_geo_ref, nd_val, grid.shape[0], grid.shape[1], xy.shape[0])
    return out


def flood_cells(dem, cut_off, water_mask):
    # experimental 'downhill' expansion of water cells
    assert(water_mask.shape == dem.shape)
    out = np.copy(water_mask)
    n = lib.flood_cells(dem, cut_off, water_mask, out,
                        dem.shape[0], dem.shape[1])
    return out, n


def masked_mean_filter(dem, mask, rad=2):
    """
    Mean filter of a dem, using only values within mask and changing only values within mask.
    """
    assert(mask.shape == dem.shape)
    assert(rad >= 1)
    out = np.copy(dem)
    lib.masked_mean_filter(dem, out, mask, rad, dem.shape[0], dem.shape[1])
    return out


def walk_mask(M, start, end):
    start = np.asarray(start, dtype=np.int32)
    end = np.asarray(end, dtype=np.int32)
    for pos in (start, end):
        assert pos.size == 2
        assert pos.max() < max(M.shape)
        assert pos.min() >= 0
        assert M[pos[0], pos[1]]
    path = np.zeros((M.size, 2), dtype=np.int32)
    N = M.copy()
    path_size = lib.walk_mask(N, start, end, path, path.shape[0], M.shape[0], M.shape[1])
    return np.resize(path, (path_size, 2))


def resample_grid(grid, nd_val, geo_ref_in, geo_ref_out, ncols_out, nrows_out):
    """
    Resample (upsample / downsample) a grid using bilinear interpolation.
    Args:
        grid: numpy input 2d array (float64)
        nd_val: output no data value
        georef: iterable of floats: (xulcenter, hor_cellsize, yulcenter, vert_cellsize). NOT GDAL style georeference.
        ncols_out: Number of columns in output.
        nrows_out: Number of rows in output.
    Returns:
        output numpy 2d array (float64)
    """
    if len(geo_ref_in) != 4 or len(geo_ref_out) != 4:
        raise Exception("Geo reference should be sequence of len 4, xulcenter, cx, yulcenter, cy")
    geo_ref_in = GEO_REF_ARRAY(*geo_ref_in)
    geo_ref_out = GEO_REF_ARRAY(*geo_ref_out)
    p_geo_ref_in = ctypes.cast(geo_ref_in, LP_CDOUBLE)  # null or pointer to geo_ref
    p_geo_ref_out = ctypes.cast(geo_ref_out, LP_CDOUBLE)  # null or pointer to geo_ref
    grid = np.require(grid, dtype=np.float64, requirements=['A', 'O', 'C', 'W'])
    out = np.empty((nrows_out, ncols_out), dtype=np.float64)
    lib.resample_grid(
        grid,
        out,
        p_geo_ref_in,
        p_geo_ref_out,
        nd_val,
        grid.shape[0],
        grid.shape[1],
        nrows_out,
        ncols_out)
    return out


# slow, but flexible method designed to calc. some algebraic quantity of q's within every single cell
def make_grid(xy, q, ncols, nrows, georef, nd_val=-9999, method=np.mean, dtype=None, srs=None):  # gdal-style georef
    """
    Apply a function on scattered data (xy) to produce a regular grid.
    Will apply the supplied method on the points that fall within each output cell.
    Args:
        xy: numpy array of shape (n,2).
        q: 1d numpy array. The value to 'grid'.
        ncols: Number of columns in output.
        nrows: Number of rows in output.
        georef: GDAL style georeference (list / tuple containing 6 floats).
        nd_val: Output no data value.
        method: The method to apply to the points that are contained in each cell.
        dtype: Output numpy data type.
    Returns:
        2d numpy array of shape (nrows,ncols).
    """
    if dtype is None:
        dtype = q.dtype
    out = np.ones((nrows, ncols), dtype=dtype) * nd_val
    arr_coords = ((xy - (georef[0], georef[3])) / (georef[1], georef[5])).astype(np.int32)
    M = np.logical_and(arr_coords[:, 0] >= 0, arr_coords[:, 0] < ncols)
    M &= np.logical_and(arr_coords[:, 1] >= 0, arr_coords[:, 1] < nrows)
    arr_coords = arr_coords[M]
    q = q[M]
    # create flattened index
    B = arr_coords[:, 1] * ncols + arr_coords[:, 0]
    # now sort array
    I = np.argsort(B)
    arr_coords = arr_coords[I]
    q = q[I]
    # and finally loop through pts just one more time...
    box_index = arr_coords[0, 1] * ncols + arr_coords[0, 0]
    i0 = 0
    row = arr_coords[0, 1]
    col = arr_coords[0, 0]
    for i in xrange(arr_coords.shape[0]):
        b = arr_coords[i, 1] * ncols + arr_coords[i, 0]
        if (b > box_index):
            # set the current cell
            out[row, col] = method(q[i0:i])
            # set data for the next cell
            i0 = i
            box_index = b
            row = arr_coords[i, 1]
            col = arr_coords[i, 0]
    # set the final cell - corresponding to largest box_index
    assert ((arr_coords[i0] == arr_coords[-1]).all())
    final_val = method(q[i0:])
    out[row, col] = final_val
    return Grid(out, georef, nd_val, srs=srs)


def grid_most_frequent_value(xy, q, ncols, nrows, georef, v1=None, v2=None, nd_val=-9999, srs=None):
    """
    Grid the most frequent value (q) for the points (xy) that fall within each cell.
    Grid extent specified via ncols, nrows and GDAL style georeference.
    """
    # void grid_most_frequent_value(int *sorted_indices, int *values, int
    # *out, int vmin,int vmax,int nd_val, int n)
    out = np.ones((nrows, ncols), dtype=np.int32) * nd_val
    arr_coords = ((xy - (georef[0], georef[3])) / (georef[1], georef[5])).astype(np.int32)
    M = np.logical_and(arr_coords[:, 0] >= 0, arr_coords[:, 0] < ncols)
    M &= np.logical_and(arr_coords[:, 1] >= 0, arr_coords[:, 1] < nrows)
    arr_coords = arr_coords[M]
    q = q[M]
    # create flattened index
    B = arr_coords[:, 1] * ncols + arr_coords[:, 0]
    del arr_coords
    # now sort array
    I = np.argsort(B)
    B = B[I]
    q = q[I]
    if v1 is None:
        v1 = q.min()
    if v2 is None:
        v2 = q.max()
    size = (v2 - v1 + 1)
    if size < 1 or size > 20000:
        raise ValueError("Invalid range: %d" % size)
    lib.grid_most_frequent_value(B, q, out, v1, v2, nd_val, B.shape[0])
    return Grid(out, georef, nd_val, srs=srs)


def user2array(georef, xy):
    # Return array coordinates (as int32 here) for input points in 'real'
    # coordinates. georef is a GDAL style georeference.
    return ((xy - (georef[0], georef[3])) / (georef[1], georef[5])).astype(np.int32)


def grid_extent(geo_ref, shape):
    # Just calculate the extent of a grid.
    # use a GDAL-style georef and a numpy style shape
    x1 = geo_ref[0]
    y2 = geo_ref[3]
    x2 = x1 + shape[1] * geo_ref[1]
    y1 = y2 + shape[0] * geo_ref[5]
    return (x1, y1, x2, y2)


def intersect_grid_extents(georef1, shape1, georef2, shape2):
    # Will calculate the pixel extent (slices) of the intersection of two grid extents in each pixel coord. set
    # Avoid rounding issues
    if (georef1[1] != georef2[1] or georef1[5] != georef2[5]):  # TODO check alignment
        raise ValueError("Must have same cell size and 'alignment'")
    extent1 = np.array(grid_extent(georef1, shape1))
    extent2 = np.array(grid_extent(georef2, shape2))
    # calculate intersection
    x1, y1 = np.maximum(extent1[:2], extent2[:2])
    x2, y2 = np.minimum(extent1[2:], extent2[2:])
    # print x1,y1,x2,y2
    # print extent1
    # print extent2
    if (x1 >= x2 or y1 >= y2):
        return None, None
    ullr = np.array(((x1, y2), (x2, y1)), dtype=np.float64)
    ullr += (georef1[1] * 0.5, georef1[5] * 0.5)  # avoid rounding
    slice1 = user2array(georef1, ullr)
    slice2 = user2array(georef2, ullr)
    # will return two (row_slice,col_slices)
    rs1 = slice(slice1[0, 1], slice1[1, 1])
    rs2 = slice(slice2[0, 1], slice2[1, 1])
    cs1 = slice(slice1[0, 0], slice1[1, 0])
    cs2 = slice(slice2[0, 0], slice2[1, 0])
    return (rs1, cs1), (rs2, cs2)


def create_gdal_ds(cstr, geo_ref, data_type, shape, fmt="GTiff", nd_val=None,
                   dco=None, srs=None, fill_val=None):
    """
    Create a (1 band) GDAL raster datasource.
    Args:
       cstr: Connection string / path to datasource.
       geo_ref: GDAL style georeference (list of len 6).
       data_type: GDAL data type.
       shape: Shape of raster (nrows, ncols).
       fmt: GDAL driver name, defaults to GTiff.
       nd_val: No data value to set on band.
       dco: Dataset creation options (driver specific - refer to GDAL docs).
       srs: osr.SpatialReference, OR a string containing GDAL wkt definition.
       fill_val: Value to initialise with. If None and nd_val is not None,
                 fill_val will be nd_val.
    Returns:
       Reference to GDAL datasource.
    """
    driver = gdal.GetDriverByName(fmt)
    assert(driver is not None)
    if fmt != "MEM" and os.path.isfile(cstr):
        try:
            driver.Delete(cstr)
        except Exception as e:
            LOG.error(str(e))
        else:
            LOG.info("Overwriting %s..." % cstr)
    else:
        LOG.info("Creating %s..." % cstr)
    if dco:
        dst_ds = driver.Create(cstr, shape[1], shape[0], 1, data_type, options=dco)
    else:
        dst_ds = driver.Create(cstr, shape[1], shape[0], 1, data_type)
    dst_ds.SetGeoTransform(geo_ref)
    if srs is not None:
        if isinstance(srs, osr.SpatialReference):
            srs_wkt = srs.ExportToWkt()
        else:
            # GDAL should complain if this is something bad
            srs_wkt = srs
        dst_ds.SetProjection(srs_wkt)
    band = dst_ds.GetRasterBand(1)
    if nd_val is not None:
        band.SetNoDataValue(nd_val)
        if fill_val is None:
            fill_val = nd_val
    if fill_val is not None:
        LOG.info("Filling band with: %s" % fill_val)
        band.Fill(fill_val)
    return dst_ds


class Grid(object):
    """
    Grid abstraction class (1 band image).
    Contains a numpy array and metadata like geo reference.
    """

    def __init__(self, arr, geo_ref, nd_val=None, srs=None):
        """
        Args:
            arr: Numpy 2d array
            geo_ref: GDAL style georeference (left_edge, cx, 0, top_edge, 0, cy)
            nd_val: No data value
            srs: GDAL srs wkt (as returned by osr.SpatialReference.ExportToWkt())
        """
        self.grid = arr
        self.geo_ref = np.array(geo_ref)
        self.nd_val = nd_val
        if srs is not None and not isinstance(srs, osr.SpatialReference):
            raise TypeError("srs must be osr.SpatialReference")
        self.srs = srs
        # and then define some useful methods...

    @property
    def shape(self):
        return self.grid.shape

    @property
    def dtype(self):
        return self.grid.dtype

    def expand_vert(self, pos, buf):
        assert(self.nd_val is not None)
        band = np.ones((buf, self.grid.shape[1]), dtype=self.grid.dtype) * self.nd_val
        if pos < 0:  # top
            self.grid = np.vstack((band, self.grid))
            self.geo_ref[3] -= self.geo_ref[5] * buf
        elif pos > 0:  # bottom
            self.grid = np.vstack((self.grid, band))
        return self

    def expand_hor(self, pos, buf):
        assert(self.nd_val is not None)
        band = np.ones((self.grid.shape[0], buf), dtype=self.grid.dtype) * self.nd_val
        if pos < 0:  # left
            self.grid = np.hstack((band, self.grid))
            self.geo_ref[0] -= self.geo_ref[1] * buf
        elif pos > 0:  # right
            self.grid = np.hstack((self.grid, band))
        return self
    # shrink methods should return views - so beware... perhaps use resize...

    def shrink_vert(self, pos, buf):
        """
        Shrink the grid vertically by buf pixels. If pos>0 shrink from bottom, if pos<0 shrink from top.
        Beware: The internal grid will now be a view.
        """
        assert(self.grid.shape[0] > buf)
        if pos < 0:  # top
            self.grid = self.grid[buf:, :]
            self.geo_ref[3] += self.geo_ref[5] * buf
        elif pos > 0:  # bottom
            self.grid = self.grid[:-buf, :]
        return self

    def shrink_hor(self, pos, buf):
        """
        Shrink the grid horisontally by buf pixels. If pos>0 shrink from right, if pos<0 shrink from left.
        Beware: The internal grid will now be a view.
        """
        assert(self.grid.shape[1] > buf)
        if pos < 0:  # left
            self.grid = self.grid[:, buf:]
            self.geo_ref[0] += self.geo_ref[1] * buf
        elif pos > 0:  # right
            self.grid = self.grid[:, :-buf]
        return self

    def shrink(self, shrink, copy=False):
        # Will return a view unless copy=True, be carefull! Can be extended to
        # handle more general slices...
        assert(min(self.grid.shape) > 2 * shrink)
        if shrink <= 0:
            return self
        G = self.grid[shrink:-shrink, shrink:-shrink]
        if copy:
            G = G.copy()
        geo_ref = list(self.geo_ref[:])
        geo_ref[0] += shrink * self.geo_ref[1]
        geo_ref[3] += shrink * self.geo_ref[5]
        return Grid(G, geo_ref, self.nd_val)

    def interpolate(self, xy, nd_val=None):
        """
        Bilinear grid interpolation. Grid data type must be float64.
        Args:
            xy: Input points (numpy array of shape (n,2))
            nd_val: Output no data value. Should only be supplied if the grid does not have a no data value.
        Returns:
            1d numpy array of interpolated values
        """
        # If the grid does not have a nd_val, the user must supply one here...
        if self.nd_val is None:
            if nd_val is None:
                raise Exception("No data value not supplied...")
        else:
            if nd_val is not None:
                raise Warning("User supplied nd-val not used as grid already have one...")
            nd_val = self.nd_val
        cx = self.geo_ref[1]
        cy = self.geo_ref[5]
        # geo_ref used in interpolation ('corner' coordinates...)
        cell_georef = [self.geo_ref[0] + 0.5 * cx, cx, self.geo_ref[3] + 0.5 * cy, -cy]
        return bilinear_interpolation(self.grid, xy, nd_val, cell_georef)

    def warp(self, dst_srs, cx=None, cy=None, out_extent=None, resample_method=gdal.GRA_Bilinear):
        """
        Warp to dst_srs (given as a GDAL wkt definition)
        Args:
            dst_srs: osr.SpatialReference
            cx: horisontal output cellsize (calculate if None)
            cy: vertical output cellsize (>0, calculate if None)
            out_extent: (xmin, ymin, xmax, ymax) in output coord sys. (Will be calculated if None)
            resample_method: A supported GDAL resample method, defaults to gdal.GRA_Bilinear
        Returns:
            a new grid object
        """
        if self.srs is None:
            raise ValueError("Needs a srs definition for self.")
        if not isinstance(dst_srs, osr.SpatialReference):
            raise TypeError("srs must be osr.SpatialReference")
        src_ds = self.as_gdal_dataset("in", "MEM")
        # calculate dest georef and size
        extent = map(float, self.extent)
        source_srs = self.srs
        target_srs = dst_srs
        transform = osr.CoordinateTransformation(source_srs, target_srs)
        new_corners = ((extent[0], extent[1]), (extent[0], extent[3]),
                       (extent[2], extent[1]), (extent[2], extent[3]))
        center = (extent[0] + (extent[2] - extent[0]) * 0.5, extent[1] + (extent[3] - extent[1]) * 0.5)
        if out_extent is None:
            xmin, xmax, ymin, ymax = None, None, None, None
            for pt in new_corners:
                x, y, z = transform.TransformPoint(float(pt[0]), float(pt[1]), 0)
                xmin = min(x, xmin) if xmin is not None else x
                xmax = max(x, xmax) if xmax is not None else x
                ymin = min(y, ymin) if ymin is not None else y
                ymax = max(y, ymax) if ymax is not None else y
        else:
            xmin, ymin, xmax, ymax = out_extent
        if cx is None:
            # match same cellsize (in center)
            x1, y1, z1 = transform.TransformPoint(center[0], center[1], 0)
            x2, y2, z2 = transform.TransformPoint(center[0] + self.geo_ref[1], center[1], 0)
            cx = x2 - x1
        if cy is None:
            x1, y1, z1 = transform.TransformPoint(center[0], center[1])
            # go 'up'
            x2, y2, z2 = transform.TransformPoint(center[0], center[1] - self.geo_ref[5], 0)
            cy = y2 - y1
        assert cx > 0 and cy > 0
        new_georef = [xmin, cx, 0, ymax, 0, -cy]
        ncols = int(np.ceil((xmax - xmin) / cx))
        nrows = int(np.ceil((ymax - ymin) / cy))
        assert ncols < 5 * 1e4 and nrows < 5 * 1e4
        dst_ds = create_gdal_ds("out", new_georef, self.npy2gdaltype(self.dtype), (nrows, ncols), 
                                fmt="MEM", nd_val=self.nd_val, srs=dst_srs)
        band = dst_ds.GetRasterBand(1)
        band.WriteArray(np.ones((nrows, ncols), dtype=self.dtype) * self.nd_val)
        rc = gdal.ReprojectImage(src_ds, dst_ds, self.srs.ExportToWkt(), dst_srs.ExportToWkt(), resample_method)
        assert rc == 0
        grid_out = dst_ds.ReadAsArray()
        return Grid(grid_out, new_georef, self.nd_val, srs=dst_srs)

    @classmethod
    def npy2gdaltype(cls, npy_dtype):
        """Get corresponding GDAL datatype"""
        if npy_dtype == np.float32:
            return gdal.GDT_Float32
        elif npy_dtype == np.float64:
            return gdal.GDT_Float64
        elif npy_dtype == np.int32:
            return gdal.GDT_Int32
        elif npy_dtype == np.uint32:
            return gdal.GDT_UInt32
        elif npy_dtype == np.bool or npy_dtype == np.uint8:
            return gdal.GDT_Byte
        elif npy_dtype == np.uint16:
            return gdal.GDT_Uint16
        elif npy_dtype == np.int16:
            return gdal.GDT_Int16
        else:
            raise NotImplementedError("dtype not supported - yet")

    def as_gdal_dataset(self, fname, fmt="GTiff", dco=None, srs=None):
        gdal_dtype = self.npy2gdaltype(self.dtype)
        if srs is None:  # will override self.srs which is default if set
            srs = self.srs
        if srs is not None:
            if not isinstance(srs, osr.SpatialReference):
                raise TypeError("srs must be osr.SpatialReference")
        dst_ds = create_gdal_ds(fname, self.geo_ref, gdal_dtype,
                                self.shape, fmt, self.nd_val,
                                dco, srs)
        dst_ds.GetRasterBand(1).WriteArray(self.grid)
        return dst_ds

    def save(self, fname, fmt="GTiff", dco=None, srs=None):
        dst_ds = self.as_gdal_dataset(fname, fmt, dco, srs)
        dst_ds = None  # flush

    @property
    def extent(self):
        return self.get_bounds()

    def get_bounds(self):
        """Return grid extent as (xmin, ymin, xmax, ymax)"""
        return grid_extent(self.geo_ref, self.grid.shape)

    def correlate(self, other):
        pass  # TODO

    # method 0 is Horn - smoother, otherwise Zevenberg-Thorne - faster.
    def get_hillshade(self, azimuth=315, height=45, z_factor=1.0, method=0):
        # requires scipy.ndimage
        # light should be the direction to the sun
        if not HAS_NDIMAGE:
            raise ValueError("This method requires scipy.ndimage")
        ang = np.radians(360 - azimuth + 90)
        h_rad = np.radians(height)
        light = np.array((np.cos(ang) * np.cos(h_rad), np.sin(ang) * np.cos(h_rad), np.sin(h_rad)))
        light = light / (np.sqrt(light.dot(light)))  # normalise
        if method == 0:
            kernel = H_KERNEL
            k_factor = 8
        else:
            kernel = ZT_KERNEL
            k_factor = 2
        scale_x = z_factor / (self.geo_ref[1] * k_factor)  # scale down
        scale_y = z_factor / (self.geo_ref[5] * k_factor)
        dx = image.filters.correlate(self.grid, kernel) * scale_x
        # taking care of revered axis since cy<0
        dy = image.filters.correlate(self.grid, kernel.T) * scale_y
        # The normal vector looks like (-dx,-dy,1) - in array coords: (-dx,dy,1)
        X = np.sqrt(dx ** 2 + dy ** 2 + 1)  # the norm of the normal
        # calculate the dot product and normalise - should be in range -1 to 1 -
        # less than zero means black, which here should translate to the value 1
        # as a ubyte.
        X = (-dx * light[0] - dy * light[1] + light[2]) / X
        # print X.min(), X.max()
        X[X < 0] = 0  # dark pixels should have value 1
        X = X * 254 + 1
        # should not happen
        X[X > 255] = 255  # there should be none
        # end should not happen
        M = (self.grid == self.nd_val)
        if M.any():
            T = ((np.fabs(kernel) + np.fabs(kernel.T)) > 0)
            M = image.morphology.binary_dilation(M, T)
        X[M] = 0
        X = X.astype(np.uint8)
        return Grid(X, self.geo_ref, nd_val=0, srs=self.srs)  # cast shadow
