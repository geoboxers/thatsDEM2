# Original work Copyright (c) 2015, Danish Geodata Agency <gst@gst.dk>
# Modified work Copyright (c) 2015, 2016, Geoboxers <info@geoboxers.com>
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

############################
# Pointcloud utility class - wraps many useful methods
# silyko, 2014 - 2016
############################

import os
import ctypes
import numpy as np
from math import ceil
from osgeo import gdal, ogr
import thatsDEM2.triangle as triangle
import thatsDEM2.array_geometry as array_geometry
import thatsDEM2.vector_io as vector_io
# Should perhaps be moved to method in order to speed up import...
import thatsDEM2.grid as grid
import thatsDEM2.remote_files as remote_files
# Import las reader modules
try:
    import laspy.file
except ImportError:
    HAS_LASPY = False
else:
    HAS_LASPY = True
try:
    import slash
except Exception:
    HAS_SLASH = False
else:
    HAS_SLASH = True


class InvalidArrayError(Exception):
    pass


# Translation of short lidar attr codes to laspy names
LASPY_ATTRS = {"c": "raw_classification",
               "pid": "pt_src_id",
               "rn": "return_num",
               "i": "intensity"}


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



def empty_like(pc):
    """
    Contruct and empty Pointcloud object with same attributes as input pointcloud.
    Args:
        pc: Pointcloud.pointcloud object.
    Returns:
        Pointcloud object.
    """
    out = type(pc)(np.empty((0, 2), dtype=np.float64),
                   np.empty((0,), dtype=np.float64))
    for a in pc.attributes:
        array = pc.get_array(a)
        out.set_attribute(a, np.empty((0,), dtype=array.dtype))
    return out


class Pointcloud(object):
    """
    Pointcloud class constructed from a xy and a z array.
    Additional properties given in pc_attrs must be 1d numpy arrays.
    Pointcloud properties as well as xy and z will be directly modifiable by design,
    for example like pc.xy += 1.35 and pc.c[pc.c == 4] = 5,
    but make sure you know what you're doing in order to keep consistency in sizes.
    And note that if you do direct modifications like that, derived attributes like
    triangulation, sorting and bounding box may be inconsistent - remember to clear
    with Pointcloud.clear_derived_attrs()
    """

    def __init__(self, xy, z, **pc_attrs):
        xy = self._validate_array("xy", xy, check_size=False)
        z = self._validate_array("z", z, check_size=False)
        if z.shape[0] != xy.shape[0]:
            raise ValueError("z must have length equal to number of xy-points")
        # All pointcloudy arrays, including xy and z
        self.__pc_attrs = {"xy", "z"}
        self.xy = xy
        self.z = z
        # Derived attrs
        self.clear_derived_attrs()
        # store the additional attributes
        for a in pc_attrs:
            self.set_attribute(a, pc_attrs[a])

    def __setattr__(self, name, value):
        """Try to keep consistency if pointcloud attributes are set directly"""
        # By logic shortcut we can ALWAYS add a new attribute
        # But we cannot add an attribute twice before __pc_attrs is defined!
        if name in self.__dict__ and name in self.__pc_attrs:
            try:
                self._set_array(name, value, True)
            except Exception as e:
                raise InvalidArrayError(str(e))

        else:
            object.__setattr__(self, name, value)

    def __getitem__(self, i):
        """Return a dict with values at a specific index"""
        return {a: self.get_array(a)[i] for a in self.__pc_attrs}

    def astype(self, subclass):
        """
        Return data as a subclass.
        subclass must be a subclass of Pointcloud.
        """
        if not issubclass(subclass, Pointcloud):
            raise ValueError("Not a Pointcloud subclass")
        new_instance = subclass(self.xy, self.z)
        for a in self.attributes:
            new_instance.set_attribute(a, self.get_array(a))
        return new_instance

    def copy(self):
        """
        Return a copy of self as a new instance.
        """
        return self.astype(self.__class__)

    @property
    def attributes(self):
        """Return the attributes minus xy and z"""
        return self.__pc_attrs.difference({"xy", "z"})

    def set_attribute(self, name, value):
        """
        Set or add a an additional pointcloud attribute.
        Args:
            name: name of attribute
            value: array like, must have dimension 1 and same size as self.
        """
        if name in ("xy", "z"):
            raise ValueError(
                "Name of an additional attribute cannot be xy or z")

        self.set_array(name, value)

    def _validate_array(self, name, value, check_size=True):
        """Do the array checking stuff for all arrays:
        xy, z as well as additional attributes"""

        value = np.asarray(value)
        if name == "xy" or name == "z":
            value = np.require(value, requirements=[
                               'A', 'O', 'C'], dtype=np.float64)
        else:
            value = np.require(value, requirements=['A', 'O', 'C'])
        if check_size:
            assert value.shape[0] == self.xy.shape[0]
        if name != "xy":
            assert value.ndim == 1
        else:
            assert value.ndim == 2

        return value

    def set_array(self, name, array):
        """A method to do the array checking stuff and
        set array for all pointcloudy arrays, including xy and z"""
        self._set_array(name, array, True)

    def _set_array(self, name, array, size_check=False):
        """Internal version of set array, with no size checks"""
        # Unless someone tampers with __pc_attrs or deletes attributes,
        # there should be consistency
        # between names in __pc_attrs and attributes of self
        array = self._validate_array(name, array, size_check)
        self.__pc_attrs.add(name)
        object.__setattr__(self, name, array)

    def get_array(self, name):
        if name in self.__pc_attrs:
            return self.__dict__[name]
        raise ValueError("Pointcloud does not have %s attribute" % name)

    def remove_attribute(self, name):
        if name in self.attributes:
            delattr(self, name)
            self.__pc_attrs.remove(name)

    def get_unique_attribute_values(self, name):
        if name in self.attributes:
            return np.unique(self.get_array(name))
        raise ValueError("Pointcloud does not have %s attribute" % name)

    def extend(self, other, least_common=False):
        """
        Extend the pointcloud 'in place' by adding another pointcloud.
        Attributtes of current pointcloud must be a subset of attributes of other.
        Args:
            other: A pointcloud.Pointcloud object
            least_common: Whether to restrict to least common set of attributes.
        Raises:
            ValueError: If other pointcloud does not have at least the same attributes as self.
        """
        if not isinstance(other, Pointcloud):
            raise ValueError("Other argument must be a Pointcloud")
        common = self.attributes.intersection(other.attributes)
        additional = self.attributes.difference(common)
        if len(additional) > 0:
            if not least_common:
                raise ValueError(
                    "Other pointcloud does not have all attributes of self.")
            # else delete additional
            for a in additional:
                self.remove_attribute(a)
        self.clear_derived_attrs()
        for a in self.__pc_attrs:
            # Will not invoke __setattr__
            self._set_array(a, np.concatenate(
                (self.get_array(a), other.get_array(a))))

    def thin(self, I):
        """
        Modify the pointcloud 'in place' by slicing to a mask or index array.
        Args:
            I: Mask, index array (1d) or slice to use for fancy numpy indexing.
        """
        # Modify in place
        self.clear_derived_attrs()
        for a in self.__pc_attrs:
            self._set_array(a, self.get_array(a)[I])

    def cut(self, mask):
        """
        Cut the pointcloud by a mask or index array using fancy indexing.
        Args:
            mask: Mask or index array (1d) to use for fancy numpy indexing.
        Returns:
            The 'sliced' Pointcloud object.
        """
        if self.xy.size == 0:  # just return something empty to protect chained calls...
            return empty_like(self)
        pc = type(self)(self.xy[mask], self.z[mask])
        for a in self.attributes:
            pc.set_attribute(a, self.get_array(a)[mask])
        return pc

    def sort_spatially(self, cs, shape=None, xy_ul=None, keep_sorting=False):
        """
        Primitive spatial sorting by creating a 'virtual' 2D grid covering the pointcloud
        and thus a 1D index by consecutive c style numbering of cells.
        Keep track of 'slices' of the pointcloud within each 'virtual' cell.
        As the pointcloud is reordered all derived attributes will be cleared.
        Returns:
            A reference to self.
        """

        if self.get_size() == 0:
            raise Exception("No way to sort an empty pointcloud.")
        if (bool(shape) != bool(xy_ul)):  # either both None or both given
            raise ValueError(
                "Neither or both of shape and xy_ul should be specified.")
        self.clear_derived_attrs()
        if shape is None:
            x1, y1, x2, y2 = self.get_bounds()
            ncols = int((x2 - x1) / cs) + 1
            nrows = int((y2 - y1) / cs) + 1
        else:
            x1, y2 = xy_ul
            nrows, ncols = shape
        arr_coords = ((self.xy - (x1, y2)) / (cs, -cs)).astype(np.int32)
        # do we cover the whole area?
        mx, my = arr_coords.min(axis=0)
        Mx, My = arr_coords.max(axis=0)
        assert(min(mx, my) >= 0 and Mx < ncols and My < nrows)
        B = arr_coords[:, 1] * ncols + arr_coords[:, 0]
        I = np.argsort(B)
        B = B[I]
        self.thin(I)  # This will clear derived attrs
        # fix attr setting order - call thin later...
        self.spatial_index = np.ones((ncols * nrows * 2,), dtype=np.int32) * -1
        res = array_geometry.lib.fill_spatial_index(
            B, self.spatial_index, B.shape[0], ncols * nrows)
        if res != 0:
            raise Exception(
                "Size of spatial index array too small! Programming error!")
        if keep_sorting:
            self.sorting_indices = I
        self.index_header = np.asarray(
            (ncols, nrows, x1, y2, cs), dtype=np.float64)
        return self

    def sort_back(self):
        """If pc is sorted, sort it back... in place ....
        """
        if self.sorting_indices is not None:
            I = np.argsort(self.sorting_indices)
            self.thin(I)
        else:
            raise ValueError("No sorting indices")

    def clear_derived_attrs(self):
        """
        Clear derived attributes which will change after an in place modification, like an extension.
        """
        # Clears attrs which become invalid by an extentsion or sorting
        self.triangulation = None
        self.index_header = None
        self.spatial_index = None
        self.bbox = None
        self.triangle_validity_mask = None
        self.sorting_indices = None

    def might_overlap(self, other):
        return self.might_intersect_box(other.get_bounds())

    def might_intersect_box(self, box):  # box=(x1,y1,x2,y2)
        if self.xy.shape[0] == 0 or box is None:
            return False
        b1 = self.get_bounds()
        xhit = box[0] <= b1[0] <= box[2] or b1[0] <= box[0] <= b1[2]
        yhit = box[1] <= b1[1] <= box[3] or b1[1] <= box[1] <= b1[3]
        return xhit and yhit

    # Properties - nice shortcuts
    @property
    def bounds(self):
        return self.get_bounds()

    @property
    def size(self):
        return self.get_size()

    @property
    def z_bounds(self):
        return self.get_z_bounds()

    @property
    def extent(self):
        if self.xy.shape[0] > 0:
            bbox = self.get_bounds()
            z1, z2 = self.get_z_bounds()
            extent = np.zeros((6,), dtype=np.float64)
            extent[0:2] = bbox[0:2]
            extent[3:5] = bbox[2:4]
            extent[2] = z1
            extent[5] = z2
            return extent
        return None

    def get_bounds(self):
        """Return planar bounding box as (x1,y1,x2,y2) or None if empty."""
        if self.bbox is None:
            if self.xy.shape[0] > 0:
                self.bbox = array_geometry.get_bounds(self.xy)
            else:
                return None
        return self.bbox

    def get_z_bounds(self):
        """Return z bounding box as (z1,z2) or None if empty."""
        if self.z.size > 0:
            return np.min(self.z), np.max(self.z)
        else:
            return None

    def get_size(self):
        """Return point count."""
        return self.xy.shape[0]

    def cut_to_polygon(self, rings):
        """
        Cut the pointcloud to a polygon.
        Args:
            rings: list of rings as numpy arrays.
                   The first entry is the outer ring, while subsequent are holes. Holes in holes not supported.
        Returns:
            A new Pointcloud object.
        """
        I = array_geometry.points_in_polygon(self.xy, rings)
        return self.cut(I)

    def cut_to_line_buffer(self, vertices, dist):
        """
        Cut the pointcloud to a buffer around a line (quite fast).
        Args:
            vertices: The vertices of the line string as a (n,2) float64 numpy array.
            dist: The buffer distance.
        Returns:
            A new Pointcloud object.
        """
        I = array_geometry.points_in_buffer(self.xy, vertices, dist)
        return self.cut(I)

    def cut_to_box(self, xmin, ymin, xmax, ymax):
        """Cut the pointcloud to a planar bounding box"""
        I = np.logical_and((self.xy >= (xmin, ymin)),
                           (self.xy <= (xmax, ymax))).all(axis=1)
        return self.cut(I)

    def get_grid_mask(self, M, georef):
        """
        Get the boolean mask indicating which points lie within a (nrows,ncols) mask.
        Args:
            M: A numpy boolean array of shape (nrows,ncols).
            georef: The GDAL style georefence of the input mask.
        Returns:
            A numpy 1d boolean mask.
        """
        ac = ((self.xy - (georef[0], georef[3])) /
              (georef[1], georef[5])).astype(np.int32)
        N = np.logical_and(ac >= (0, 0), ac < (
            M.shape[1], M.shape[0])).all(axis=1)
        ac = ac[N]
        MM = np.zeros((self.xy.shape[0],), dtype=np.bool)
        MM[N] = M[ac[:, 1], ac[:, 0]]
        return MM

    def cut_to_grid_mask(self, M, georef):
        """
        Cut to the which points lie within a (nrows,ncols) mask.
        Args:
            M: A numpy boolean array of shape (nrows,ncols).
            georef: The GDAL style georefence of the input mask.
        Returns:
            A new Pontcloud object.
        """
        MM = self.get_grid_mask(M, georef)
        return self.cut(MM)

    def cut_to_z_interval(self, zmin, zmax):
        """
        Cut the pointcloud to points in a z interval.
        Args:
            zmin: minimum z
            zmax: maximum z
        Returns:
            New Pointcloud object
        """
        I = np.logical_and((self.z >= zmin), (self.z <= zmax))
        return self.cut(I)

    def triangulate(self):
        """
        Triangulate the pointcloud. Will do nothing if triangulation is already calculated.
        Raises:
            ValueError: If not at least 3 points in pointcloud
        """
        if self.triangulation is None:
            if self.xy.shape[0] > 2:
                self.triangulation = triangle.Triangulation(self.xy)
            else:
                raise ValueError("Less than 3 points - unable to triangulate.")

    def set_validity_mask(self, mask):
        """
        Explicitely set a triangle validity mask.
        Args:
            mask: A boolean numpy array of size the number of triangles.
        Raises:
            ValueError: If triangulation not created or mask of inproper shape.
        """
        if self.triangulation is None:
            raise ValueError("Triangulation not created yet!")
        if mask.shape[0] != self.triangulation.ntrig:
            raise ValueError("Invalid size of triangle validity mask.")
        self.triangle_validity_mask = mask

    def clear_validity_mask(self):
        """Clear the triangle validity mask (set it to None)"""
        self.triangle_validity_mask = None

    def calculate_validity_mask(self, max_angle=45, tol_xy=2, tol_z=1):
        """
        Calculate a triangle validity mask from geometry constrains.
        Args:
            max_angle: maximal angle/slope in degrees.
            tol_xy: maximal size of xy bounding box.
            tol_z: maximal size of z bounding box.
        """
        tanv2 = np.tan(max_angle * np.pi / 180.0) ** 2  # tanv squared
        geom = self.get_triangle_geometry()
        self.triangle_validity_mask = (
            geom < (tanv2, tol_xy, tol_z)).all(axis=1)

    def get_validity_mask(self):
        # just return the validity mask
        return self.triangle_validity_mask

    def get_grid(self, ncols=None, nrows=None, x1=None, x2=None, y1=None, y2=None,
                 cx=None, cy=None, nd_val=-999, method="triangulation", attr="z", srad=None):
        """
        Grid (an attribute of) the pointcloud.
        Will calculate grid size and georeference from supplied input (or pointcloud extent).
        Args:
            ncols: number of columns.
            nrows: number of rows.
            x1: left pixel corner/edge (GDAL style).
            x2: right pixel corner/edge (GDAL style).
            y1: lower pixel corner/edge (GDAL style).
            y2: upper pixel corner/edge (GDAL style).
            cx: horisontal cell size.
            cy: vertical cell size.
            nd_val: grid no data value.
            method: One of the supported method names:
                    triangulation, return_triangles, cellcount, most_frequent,
                    idw_filter, mean_filter, max_filter, min_filter, median_filter, var_filter,
                    density_filter
                    or:
                       A callable which accepts a numpy array and returns a scalar.
                       The latter will execute the callable on the subarray of values within each cell.
            attr: The attribute to grid - defaults to z.
                  Will cast attr to float64 for triangulation method, and int 32 for most_frequent.
            srad: The search radius to use for the filter variant methods.
        Returns:
            A grid.Grid object and a grid.Grid object with triangle sizes if 'return_triangles' is specified.
        Raises:
            ValueError: If unable to calculate grid size or location from supplied input,
                        or using triangulation and triangulation not calculated or supplied with invalid method name.
        """
        # x1 = left 'corner' of "pixel", not center.
        # y2 = upper 'corner', not center.
        # TODO: Fix surprises in the logic below!!!!!
        if x1 is None:
            bbox = self.get_bounds()
            x1 = bbox[0]
        if x2 is None:
            bbox = self.get_bounds()
            x2 = bbox[2]
        if y1 is None:
            bbox = self.get_bounds()
            y1 = bbox[1]
        if y2 is None:
            bbox = self.get_bounds()
            y2 = bbox[3]
        if ncols is None and cx is None:
            raise ValueError("Unable to compute grid extent from input data")
        if nrows is None and cy is None:
            raise ValueError("Unable to compute grid extent from input data")
        if ncols is None:
            ncols = int(ceil((x2 - x1) / cx))
        else:
            assert cx is None
            cx = (x2 - x1) / float(ncols)
        if nrows is None:
            nrows = int(ceil((y2 - y1) / cy))
        else:
            assert cy is None
            cy = (y2 - y1) / float(nrows)
        # geo ref gdal style...
        geo_ref = [x1, cx, 0, y2, 0, -cy]

        if method in ("triangulation", "return_triangles"):
            if self.triangulation is None:
                raise ValueError("Create a triangulation first...")
            val = np.require(self.get_array(attr), dtype=np.float64)
            if method == "triangulation":
                g = self.triangulation.make_grid(
                    val, ncols, nrows, x1, cx, y2, cy, nd_val, return_triangles=False)
                return grid.Grid(g, geo_ref, nd_val)
            else:
                g, t = self.triangulation.make_grid(
                    val, ncols, nrows, x1, cx, y2, cy, nd_val, return_triangles=True)
            return grid.Grid(g, geo_ref, nd_val), grid.Grid(t, geo_ref, nd_val)
        elif method == "cellcount":  # density grid
            arr_coords = ((self.xy - (geo_ref[0], geo_ref[3])) /
                          (geo_ref[1], geo_ref[5])).astype(np.int32)
            M = np.logical_and(arr_coords[:, 0] >= 0, arr_coords[:, 0] < ncols)
            M &= np.logical_and(
                arr_coords[:, 1] >= 0, arr_coords[:, 1] < nrows)
            arr_coords = arr_coords[M]
            # Wow - this gridding is sooo simple! and fast!
            # create flattened index
            B = arr_coords[:, 1] * ncols + arr_coords[:, 0]
            bins = np.arange(0, ncols * nrows + 1)
            h, b = np.histogram(B, bins)
            h = h.reshape((nrows, ncols))
            return grid.Grid(h, geo_ref, 0)  # zero always nodata value here...
        elif method == "most_frequent":
            # define method which takes the most frequent value in a cell...
            # could be only mean...
            val = np.require(self.get_array(attr), dtype=np.int32)
            g = grid.grid_most_frequent_value(
                self.xy, val, ncols, nrows, geo_ref, nd_val=nd_val)
            return g
        elif method in ("density_filter", "idw_filter", "max_filter",
                        "min_filter", "mean_filter", "median_filter", "var_filter"):
            if self.spatial_index is None:
                raise ValueError("Sort pointcloud first")
            if srad is None:
                srad = self.index_header[4]
            filter_func = getattr(self, method)
            assert hasattr(filter_func, "__call__")
            pts = mesh_as_points((nrows, ncols), geo_ref)
            if method == "density_filter":
                z = filter_func(srad, xy=pts).reshape((nrows, ncols))
            else:
                z = filter_func(srad, xy=pts, nd_val=nd_val,
                                attr=attr).reshape((nrows, ncols))
            return grid.Grid(z, geo_ref, nd_val)
        elif hasattr(method, "__call__"):
            val = self.get_array(attr)
            return grid.make_grid(self.xy, val, ncols, nrows, geo_ref, nd_val=nd_val, method=method)
        else:
            raise ValueError("Unsupported method.")

    def find_triangles(self, xy_in, mask=None):
        """
        Find the (valid) containing triangles for an array of points.
        Args:
            xy_in: Numpy array of points ( shape (n,2), dtype float64)
            mask: optional triangle validity mask.
        Returns:
            Numpy array of triangle indices where -1 signals no (valid) triangle.
        """
        if self.triangulation is None:
            raise Exception("Create a triangulation first...")
        xy_in = self._validate_array("xy", xy_in, False)
        # -2 indices signals outside triangulation, -1 signals invalid, else valid
        return self.triangulation.find_triangles(xy_in, mask)

    def find_appropriate_triangles(self, xy_in, mask=None):
        """
        Find the (valid) containing triangles for an array of points.
        Either the internal triangle validity mask must be set or a mask must be supplied in call.
        Args:
            xy_in: Numpy array of points ( shape (n,2), dtype float64)
            mask: Optional triangle validity mask. Will use internal triangle_validity_mask if not supplied here.
        Returns:
            Numpy array of triangle indices where -1 signals no (valid) triangle.
        Raises:
            ValueError: If triangle validty mask not available.
        """
        if mask is None:
            mask = self.triangle_validity_mask
        if mask is None:
            raise ValueError("This method needs a triangle validity mask.")
        return self.find_triangles(xy_in, mask)

    def get_points_in_triangulation(self, xy_in):
        # Not really used. Cut input xy to the points that lie inside the triangulation.
        # Can be used to implement a point in polygon algorithm!
        I = self.find_triangles(xy_in)
        return xy_in[I >= 0]

    def get_points_in_valid_triangles(self, xy_in, mask=None):
        # Not really used. Cut input xy to the points that lie inside valid triangules.
        # Can be used to implement a point in polygon algorithm!
        I = self.find_appropriate_triangles(xy_in, mask)
        return xy_in[I >= 0]

    def get_boundary_vertices(self, M_t, M_p):
        # Experimental and not really used.
        # Find the vertices which are marked by M_p and inside triangles marked
        # by M_t
        M_out = array_geometry.get_boundary_vertices(
            M_t, M_p, self.triangulation.vertices)
        return M_out

    def interpolate(self, xy_in, nd_val=-999, mask=None):
        """
        TIN interpolate values in input points.
        Args:
            xy_in: The input points (anything that is convertable to a numpy (n,2) float64 array).
            nd_val: No data value for points not in any (valid) triangle.
            mask: Optional triangle validity mask.
        Returns:
            1d numpy array of interpolated values.
        Raises:
            ValueError: If triangulation not created.
        """
        if self.triangulation is None:
            raise ValueError("Create a triangulation first...")
        xy_in = self._validate_array("xy", xy_in, False)
        return self.triangulation.interpolate(self.z, xy_in, nd_val, mask)
    # Interpolates points in valid triangles

    def controlled_interpolation(self, xy_in, mask=None, nd_val=-999):
        """
        TIN interpolate values in input points using only valid triangles.
        Args:
            xy_in: The input points (anything that is convertable to a numpy (n,2) float64 array).
            nd_val: No data value for points not in any (valid) triangle.
            mask: Optional triangle validity mask.
                  Will use internal triangle_validity_mask if  mask not supplied in call.
        Returns:
            1d numpy array of interpolated values.
        Raises:
            ValueError: If triangulation not created or mask not available.
        """
        if mask is None:
            mask = self.triangle_validity_mask
        if mask is None:
            raise ValueError("This method needs a triangle validity mask.")
        return self.interpolate(xy_in, nd_val, mask)

    def get_triangle_geometry(self):
        """
        Calculate the triangle geometry as an array with rows: (tanv2_i,bb_xy_i,bb_z_i).
        Here tanv2 is the squared tangent of the slope angle,
        bb_xy is the maximal edge of the planar bounding box, and bb_z_i the size of the vertical bounding box.
        Returns:
            Numpy array of shape (n,3) containing the geometry numbers for each triangle in the triangulation.
        Raises:
            ValueError: If triangulation not created.
        """
        if self.triangulation is None:
            raise ValueError("Create a triangulation first...")
        return array_geometry.get_triangle_geometry(
            self.xy, self.z, self.triangulation.vertices, self.triangulation.ntrig)

    def warp(self, sys_in, sys_out):
        pass  # TODO - use TrLib

    def affine_transformation(self, R=None, T=None):
        """
        Modify points in place by an affine transformation xyz=R*xyz+T.
        Args:
            R: 3 times 3 array (scaling, rotation, etc.)
            T: Translation vector (dx,dy,dz)
        """
        # Wasting a bit of memory here to keep it simple!
        self.clear_derived_attrs()
        xyz = np.column_stack((self.xy, self.z))
        if R is not None:
            xyz = np.dot(R, xyz.T).T
        if T is not None:
            xyz += T
        self.xy = xyz[:, :2].copy()
        self.z = xyz[:, 2].copy()

    def affine_transformation_2d(self, R=None, T=None):
        """
        Modify xy points in place by a 2d affine transformation xy=R*xy+T.
        Args:
            R: 2 times 2 array (scaling, rotation, etc.)
            T: Translation vector (dx,dy)
        """
        self.clear_derived_attrs()
        if R is not None:
            self.xy = (np.dot(R, self.xy.T).T).copy()
        if T is not None:
            self.xy += T

    def toE(self, geoid):
        """
        Warp to ellipsoidal heights. Modify z 'in place' by adding geoid height.
        Args:
            geoid: A geoid grid - grid.Grid instance.
        """
        # warp to ellipsoidal heights
        toE = geoid.interpolate(self.xy)
        assert((toE != geoid.nd_val).all())
        self.z += toE

    def toH(self, geoid):
        """
        Warp to orthometric heights. Modify z 'in place' by subtracting geoid height.
        Args:
            geoid: A geoid grid - grid.Grid instance.
        """
        # warp to orthometric heights. z bounds not stored, so no need to
        # recalculate.
        toE = geoid.interpolate(self.xy)
        assert((toE != geoid.nd_val).all())
        self.z -= toE

    # dump methods
    def dump_ogr_layer(self, layer):
        # Layer must have fields corresponding to self.attributes
        geom_type = layer.GetGeomType()
        if not geom_type == ogr.wkbPoint25D:
            # TODO: handle wkbPoint
            raise TypeError("Geometry type must be wkbPoint25D")
        layer_defn = layer.GetLayerDefn()
        converter_list = []
        for a in self.attributes:
            try:
                t = layer_defn.GetFieldDefn(
                    layer_defn.GetFieldIndex(a)).GetType()
                assert t == vector_io.npytype2ogrtype(self.get_array(a).dtype)
            except Exception as e:
                raise TypeError("Layer does not seem to have a proper field corresponding to '%s'" % a +
                                "\n" + str(e))
            converter_list.append((a, float if t == ogr.OFTReal else int))
        for row in self:
            feature = ogr.Feature(layer_defn)
            geom = ogr.Geometry(geom_type)
            geom.SetPoint(0, float(row["xy"][0]), float(
                row["xy"][1]), float(row["z"]))
            for field_name, converter in converter_list:
                val = converter(row[field_name])
                feature.SetField(field_name, val)
            feature.SetGeometry(geom)
            ok = layer.CreateFeature(feature)
            assert ok == 0

    def dump_new_ogr_layer(self, ds, layername="pointcloud", srs=None):
        geom_type = ogr.wkbPoint25D
        layer = ds.CreateLayer(layername, srs, geom_type)
        field_list = [(a, vector_io.npytype2ogrtype(self.get_array(a).dtype))
                      for a in self.attributes]
        for field_name, field_type in field_list:
            field_defn = ogr.FieldDefn(field_name, field_type)
            ok = layer.CreateField(field_defn)
            assert ok == 0
        self.dump_ogr_layer(layer)
        layer = None

    def dump_new_ogr_datasource(self, cstr, fmt="ESRI Shapefile", layername="pointcloud", dsco=None, srs=None):
        drv = ogr.GetDriverByName(fmt)
        assert drv is not None
        if not dsco:
            dsco = []
        ds = drv.CreateDataSource(cstr, dsco)
        assert ds is not None
        self.dump_new_ogr_layer(ds, layername, srs)
        ds = None

    def dump_txt(self, path):
        """Just dump the xyz attrs of a pointcloud as a whitespace separated text file."""
        xyz = np.column_stack((self.xy, self.z))
        np.savetxt(path, xyz)

    def dump_npy(self, path):
        """Dump just xy and z (stacked) as a npy-file"""
        xyz = np.column_stack((self.xy, self.z))
        np.save(path, xyz)

    def dump_npz(self, path, compressed=False):
        """Dump the full pointcloud, including all attrs as a npz-file.
        If compressed is True, store in compressed format using savez_compressed
        """
        if compressed:
            np.savez_compressed(path, **{a: self.get_array(a)
                                         for a in self.__pc_attrs})
        else:
            np.savez(path, **{a: self.get_array(a) for a in self.__pc_attrs})
    
    # Class constructers
    @classmethod
    def from_npz(cls, path, attrs=None, **kwargs):
        """
        Load a pointcloud from a platform independent numpy .npz file.
        Restoring attributes by keys is supported.
        Args:
            path: path to .npz file.
            attrs: A set of attrs to return, or None - meaning all attrs.
        Returns:
            A Pointcloud - or subclass - object 
        """
        npzfile = np.load(path)
        assert "xy" in npzfile.files
        assert "z" in npzfile.files
        _attrs = set(npzfile.files).difference({"xy", "z"})
        if attrs is not None:
            _attrs = _attrs.intersection(attrs)
        pc = cls(npzfile["xy"], npzfile["z"], **{a: npzfile[a] for a in _attrs})
        return pc

    @classmethod
    def from_npy(cls, path, **kwargs):
        """
        Load a pointcloud from a platform independent numpy .npy file. Will only keep xyz.
        Args:
            path: path to .npy file.
        Returns:
            A Pointcloud - or subclass - object.
        """
        xyz = np.load(path)
        return cls(xyz[:, 0:2], xyz[:, 2])

    # make a pointcloud form an OGR readable point datasource.
    # TODO: handle multipoint features....
    @classmethod
    def from_ogr(cls, cstr, layername=None, layersql=None, extent=None, **kwargs):
        """
        Load a pointcloud from an OGR 3D-point datasource.
        Args:
            path:  OGR connection string.
            layername: name of layer to load. Use either layername or layersql.
            layersql: sql command to execute to fetch geometries. Use either layername or layersql.
            extent: Extent to filter the result set by.
        Returns:
            A Pointcloud - or subclass - object
        """
        ds, layer = vector_io.open(cstr, layername, layersql, extent)
        layer_defn = layer.GetLayerDefn()
        geom_type = layer.GetGeomType()
        assert geom_type in [ogr.wkbPoint, ogr.wkbPoint25D]
        # Determine what attributes to keep...
        attr_types = {}
        for field in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(field)
            name = field_defn.GetName()
            ogr_type = field_defn.GetType()
            try:
                dtype = vector_io.ogrtype2npytype(ogr_type)
            except Exception:
                continue
            attr_types[name] = dtype
        attrs = {name: [] for name in attr_types}
        xy = []
        z = []
        # Now do the long loop over features
        for feat in layer:
            geom = feat.GetGeometryRef()
            xy.append(geom.GetPoint_2D(0))
            for a in attrs:
                attrs[a].append(feat[a])
            if geom.GetCoordinateDimension() == 3:
                z.append(geom.GetZ(0))
        if not z:
            assert "z" in attrs
            z = attrs.pop("z")
        if layersql:
            ds.ReleaseResultSet(layer)
        layer = None
        ds = None
        # not to double up memory consumption - make conversion here, rather
        # than in the pointlcoud constructor
        xy = np.asarray(xy, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        for a in attrs:
            attrs[a] = np.asarray(attrs[a], dtype=attr_types[a])
        return cls(xy, z, **attrs)

    # make a (geometric) pointcloud from a (xyz) text file
    @classmethod
    def from_text(cls, path, delim=None, **kwargs):
        """
        Load a pointcloud (xyz) from a delimited text file.
        Args:
            path: path to file.
            delim: Delimiter, None corresponds to white space.
        Returns:
            A pointcloud.Pointcloud object containg only the raw x,y and z coords.
        """
        points = np.loadtxt(path, delimiter=delim)
        if points.ndim == 1:  # Just a single point?
            points = points.reshape((1, 3))
        return cls(points[:, :2], points[:, 2])

    @classmethod
    def from_array(cls, z, geo_ref, nd_val=None):
        """
        Construct a Pointcloud object corresponding to the cell centers of an in memory grid.
        Args:
            z: Numpy array of shape (nrows,ncols).
            geo_ref: GDAL style georefence.
            nd_val: No data value of grid. Cell centers with this value will be excluded.
        Returns:
            A Pointcloud -or subclass- object
        """
        xy = mesh_as_points(z.shape, geo_ref)
        z = z.flatten()
        if nd_val is not None:
            M = (z != nd_val)
            if not M.all():
                xy = xy[M]
                z = z[M]
        return cls(xy, z)

    @classmethod
    def from_grid(cls, path, **kwargs):
        """
        Construct a Pointcloud object corresponding to the cell centers of the first band of a GDAL loadable raster.
        Args:
            path: GDAL connection string.
        Returns:
            A Pointcloud -or subclass- object
        """
        ds = gdal.Open(path)
        geo_ref = ds.GetGeoTransform()
        nd_val = ds.GetRasterBand(1).GetNoDataValue()
        z = ds.ReadAsArray().astype(np.float64)
        ds = None
        return cls.from_array(z, geo_ref, nd_val)

    @classmethod
    def from_any(cls, path, **kwargs):
        """
        Load a pointcloud from a range of 'formats'. The specific 'driver' to use is decided from the filename extension.
        Can also handle remote files from s3 and http. Whether a file is remote is decided from the path prefix.
        Args:
            path: a 'connection string'
            additional keyword arguments will be passed on to the specific format handler.
        Returns:
            A Pointcloud - or subclass - object
        """
        # TODO - handle keywords properly - all methods, except fromLAS, will only
        # return xyz for now. Fix this...
        b, ext = os.path.splitext(path)
        # we could use /vsi<whatever> like GDAL to signal special handling -
        # however keep it simple for now.
        temp_file = None
        if remote_files.is_remote(path):
            temp_file = remote_files.get_local_file(path)
            path = temp_file
        try:
            if ext == ".las" or ext == ".laz":
                pc = cls.from_las(path, **kwargs)
            elif ext == ".npy":
                pc = cls.from_npy(path, **kwargs)
            elif ext == ".txt":
                pc = cls.from_text(path, **kwargs)
            elif ext == ".npz":
                pc = cls.from_npz(path, **kwargs)
            elif ext == ".tif" or ext == ".tiff" or ext == ".asc":
                pc = cls.from_grid(path, **kwargs)
            else:
                pc = cls.from_ogr(path, **kwargs)
        finally:
            if temp_file is not None and os.path.isfile(temp_file):
                os.remove(temp_file)
        return pc

    @classmethod
    def from_las(cls, path, attrs=("c", "pid"), **kwargs):
        """
        Load a pointcloud from las / laz format via slash.LasFile.
        Laz reading currently requires that laszip-cli is findable.
        Args:
            path: Path to las / laz file.
            attrs: Sequence of attributes to include.
                   Must be subset of LidarPointcloud.LIDAR_ATTRS
        Returns:
            A Pointcloud - or subclass - object.
        """
        if not set(attrs).issubset(set(LidarPointcloud.LIDAR_ATTRS)):
            raise ValueError(
                "Only attrs defined in Pointcloud.LIDAR_ATTRS allowed here.")
        if HAS_LASPY:
            return cls.from_laspy(path, attrs, **kwargs)
        elif HAS_SLASH:
            return cls.from_slash(path, attrs, **kwargs)
        else:
            raise Exception("No las reader library available.")

    @classmethod
    def from_slash(cls, path, attrs=("c", "pid"), **kwargs):
        """
        Use slash to read las /laz from path.
        Args:
            path: path to las/laz file.
            attrs: A subset of the short names in Pointcloud.LIDAR_ATTRS.
        Returns:
            a LidarPointcloud
        """
        plas = slash.LasFile(path)
        r = plas.read_records(include_return_number=("rn" in attrs))
        plas.close()
        xy = r["xy"]
        z = r["z"]
        for a in r.keys():
            if a in ("xy", "z") or r[a] is None:
                del r[a]
        return cls(xy, z, **r)

    @classmethod    
    def from_laspy(cls, path, attrs=("c", "pid"), **kwargs):
        """
        Use laspy to read a las/laz file.
        Args:
            path: path to las/laz file.
            attrs: A subset of the short names in Pointcloud.LIDAR_ATTRS.
        Returns:
            A LidarPointcloud
        """
        plas = laspy.file.File(path)
        xy = np.column_stack((plas.x, plas.y))
        plas_attrs = {a: getattr(plas, LASPY_ATTRS.get(a, a)) for a in attrs}
        pc = cls(xy, plas.z, **plas_attrs)
        plas.close()
        return pc

    # Filterering methods below...

    def validate_filter_args(self, rad):
        # internal utility - just validate a filter radius against the internal
        # spatial index.
        if self.spatial_index is None:
            raise Exception("Build a spatial index first!")
        if rad > self.index_header[4]:
            raise Warning(
                "Filter radius larger than cell size of spatial index will not catch all points!")

    def apply_filter(self, filter_rad, filter_func, xy, z, nd_val, attr, params):
        """Just apply a filter (string == builtin or a python callable)"""
        self.validate_filter_args(filter_rad)
        vals = np.require(self.get_array(attr), dtype=np.float64)
        out = array_geometry.apply_filter(xy, z, self.xy, vals,
                                          self.spatial_index, self.index_header,
                                          filter_func, filter_rad, nd_val, params)
        return out

    def apply_2d_filter(self, filter_rad, filter_func, xy=None,
                        nd_val=-9999, attr="z", params=None):
        """Apply a 2d filter along supplied xy or self.xy.
        Args:
            filter_rad: filter radius (less than sorting cell size).
            filter_func: A python callable of type array_geometry.FILTER_FUNC,
                         or a name (string) of a bultin filter.
            xy: Input points to filter along (will use self.xy if None).
            nd_val: No data value (if relevant).
            attr: Attribute name of value to perform filter on (defaults to z).
        Returns:
            1d-array of filtered values.
        """
        if xy is None:
            xy = self.xy
        return self.apply_filter(filter_rad, filter_func, xy, None, nd_val, attr, params)

    def apply_3d_filter(self, filter_rad, filter_func, xy=None, z=None,
                        nd_val=-9999, params=None):
        """Apply a 3d filter along supplied xy and z or self.xy, self.z.
        3d filters are geometric in the sense, that pointcloud z must be supplied.
        Filtering of another value (besides z) can be implemented
        by supplying a pointer to an array in params.
        Args:
            filter_rad: filter radius (less than sorting cell size).
            filter_func: A python callable of type array_geometry.FILTER_FUNC,
                         or a name (string) of a bultin filter.
            xy: Input points to filter along (will use self.xy if None).
            z: z-coord of input points.
            nd_val: No data value (if relevant).
            attr: Attribute name of value to perform filter on (defaults to z).
        Returns:
            1d-array of filtered values.
        """
        if xy is None or z is None:
            assert xy is None and z is None
            xy = self.xy
            z = self.z
        return self.apply_filter(filter_rad, filter_func, xy, z, nd_val, "z", params)

    # '2.5D' filters
    def min_filter(self, filter_rad, xy=None, nd_val=-9999, attr="z"):
        """
        Calculate minumum filter of z along self.xy or a supplied set of input points. Useful for gridding.
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: Optional list of input points to filter along. Will use self.xy if not supplied.
        Returns:
            1D array of filtered values.
        """
        return self.apply_2d_filter(filter_rad, "min_filter", xy, nd_val, attr)

    def mean_filter(self, filter_rad, xy=None, nd_val=-9999, attr="z"):
        """
        Calculate mean filter of z along self.xy or a supplied set of input points. Useful for gridding.
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: Optional list of input points to filter along. Will use self.xy if not supplied.
        Returns:
            1D array of filtered values.
        """
        return self.apply_2d_filter(filter_rad, "mean_filter", xy, nd_val, attr)

    def max_filter(self, filter_rad, xy=None, nd_val=-9999, attr="z"):
        """
        Calculate maximum filter of z along self.xy or a supplied set of input points. Useful for gridding.
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: Optional list of input points to filter along. Will use self.xy if not supplied.
        Returns:
            1D array of filtered values.
        """

        return self.apply_2d_filter(filter_rad, "max_filter", xy, nd_val, attr)

    def median_filter(self, filter_rad, xy=None, nd_val=-9999, attr="z"):
        """
        Calculate median filter of z along self.xy or a supplied set of input points. Useful for gridding.
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: Optional list of input points to filter along. Will use self.xy if not supplied.
        Returns:
            1D array of filtered values.
        """
        return self.apply_2d_filter(filter_rad, "median_filter", xy, nd_val, attr)

    def var_filter(self, filter_rad, xy=None, nd_val=-9999, attr="z"):
        """
        Calculate variance filter of z along self.xy or a supplied set of input points. Useful for gridding.
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: Optional list of input points to filter along. Will use self.xy if not supplied.
        Returns:
            1D array of filtered values.
        """
        return self.apply_2d_filter(filter_rad, "var_filter", xy, nd_val, attr)

    def idw_filter(self, filter_rad, xy=None, nd_val=-9999, attr="z"):
        """
        Calculate inverse distance weighted z values along self.xy or a supplied set of input points.
        Useful for gridding.
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: Optional list of input points to filter along. Will use self.xy if not supplied.
        Returns:
            1D array of filtered values.
        """
        return self.apply_2d_filter(filter_rad, "idw_filter", xy, nd_val, attr)

    # 'Geometric' filters
    def distance_filter(self, filter_rad, xy, nd_val=9999):
        """
        Calculate point distance filter along a supplied set of input points. Useful for gridding.
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: Optional list of input points to filter along. Supply this or get a lot of zeros!
        Returns:
            1D array of filtered values.
        """
        return self.apply_2d_filter(filter_rad, "distance_filter", xy, nd_val)

    def nearest_filter(self, filter_rad, xy, nd_val=-1):
        """
        Calculate index of nearest point along a supplied set of input points.
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: Optional list of input points to filter along. Supply this or get a lot of zeros!
        Returns:
            1D array of filtered values.
        """
        return self.apply_2d_filter(filter_rad, "nearest_filter", xy, nd_val).astype(np.int32)

    def density_filter(self, filter_rad, xy=None):
        """
        Calculate point density filter along self.xy or a supplied set of input points. Useful for gridding.
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: Optional list of input points to filter along. Will use self.xy if not supplied.
        Returns:
            1D array of filtered values.
        """
        return self.apply_2d_filter(filter_rad, "density_filter", xy, 0)

    # 3D filters
    def ballcount_filter(self, filter_rad, xy=None, z=None, nd_val=0):
        """
        Calculate number of points within a ball a radius filter_rad
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: array of input points to filter along. Will use self.xy if not supplied.
            z : array of input zs. Will use self.z if not supplied.
        Returns:
            1D array of counts.
        """
        return self.apply_3d_filter(filter_rad, "ballcount_filter", xy, z, 0).astype(np.int32)

    def ray_mean_dist_filter(self, filter_rad, xy=None, z=None):
        """
        Calculate mean distance in real projective space,
        of rays emanating from 'filter along' points.
         Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: Points to filter along (optional).
            z: Z coord of points to filter along (optional).
        Returns:
            1D array of spike indications (0 or 1).
        """
        return self.apply_3d_filter(filter_rad, "ray_mean_dist_filter", xy, z, 0)

    def mean_3d_filter(self, filter_rad, xy=None, z=None, nd_val=-9999, attr="z"):
        """
        Calculate mean of an attributte (defaults to z).
        Will calculate mean value of points in a 3d-ball of radius filter_rad.
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            xy: Points to filter along (optional).
            z: Z coord of points to filter along (optional).
        Returns:
            1D array of filtered values.
        """
        if attr != "z":
            vals = np.require(self.get_array(attr), dtype=np.float64)
            p_to_vals = vals.ctypes.data_as(ctypes.c_void_p)
        else:
            p_to_vals = None
        return self.apply_3d_filter(filter_rad, "mean_3d_filter", xy, z, nd_val, params=p_to_vals)

    def spike_filter(self, filter_rad, tanv2, zlim=0.2):
        """
        Calculate spike indicators (0 or 1) for each point .
        In order to be a spike there must be at least one other point within filter rad in each quadrant
        which satisfies:
        -- slope_angle large and dz large.
        See c implementation in array_geometry.c.
        Args:
            filter_rad: The radius of the filter. Should not be larger than cell size in spatial index (for now).
            tanv2: Tangent squared of slope angle (dy/dx)**2 parameter for spike check (lower limit).
            zlim: dz paramter for spike check (lower limit)
        Returns:
            1D array of spike indications (0 or 1).
        """
        if (tanv2 < 0 or zlim < 0):
            raise ValueError("Spike parameters must be positive!")
        # was:
        # params[0]=SQUARE(filter_rad*0.2);
            # params[1]=tanv2;
            # params[2]=zlim;
        arr_type = ctypes.c_double * 3
        params = arr_type((filter_rad * 0.2) ** 2, tanv2, zlim)
        p_params = ctypes.cast(params, ctypes.c_void_p)
        return self.apply_3d_filter(filter_rad, "spike_filter", params=p_params).astype(np.bool)


class LidarPointcloud(Pointcloud):
    """
    Subclass of pointcloud with special assumptions and shortcuts
    designed for lidar pointclouds.
    """
    # The 'standard' attributes for a LidarPointcloud
    # With short name to long
    LIDAR_ATTRS = {"c": "classification",
                   "rn": "return_number",
                   "pid": "point_source_id",
                   "i": "intensity",
                   "red": "red",
                   "green": "green",
                   "blue": "blue",
                   "sa": "scan_angle",
                   "sd": "scan_direction",
                   "t": "time"
                   }

    def __init__(self, xy, z, **pc_attrs):
        Pointcloud.__init__(self, xy, z, **pc_attrs)

    def set_class(self, c):
        """Explicitely set the class attribute to be c for all points."""
        self.set_attribute("c", np.ones(self.z.shape, dtype=np.uint8) * c)

    def cut_to_strip(self, pid):
        """
        Cut the pointcloud to the points with a specific point source id.
        Args:
            pid: The point source (strip) id to cut to.
        Returns:
            New Pointcloud object
        Raises:
            ValueError: If point source id attribute is not set.
        """
        if "pid" not in self.pc_attrs:
            raise ValueError("Point source id attribute not set")
        I = (self.pid == pid)
        return self.cut(I)

    def cut_to_class(self, c, exclude=False):
        """
        Cut the pointcloud to points of a specific class.
        Args:
            c: class (integer) or iterable of integers.
            exclude: boolean indicating whether to use c as an exclusive list (cut to the complement).
        Returns:
            A new Pontcloud object.
        Raises:
            ValueError: if class attribute is not set.
        """
        # will now accept a list or another iterable...
        if "c" not in self.attributes:
            raise ValueError("Class attribute not set.")
        try:
            cs = iter(c)
        except:
            cs = (c,)
        if exclude:
            I = np.ones((self.c.shape[0],), dtype=np.bool)
        else:
            I = np.zeros((self.c.shape[0],), dtype=np.bool)
        # TODO: use inplace operations to speed up...
        for this_c in cs:
            if exclude:
                I &= (self.c != this_c)
            else:
                I |= (self.c == this_c)
        return self.cut(I)

    def cut_to_return_number(self, rn):
        """
        Cut to points with return number rn (must have this attribute).
        Args:
            rn: return number to cut to.
        Returns:
            New Pointcloud object.
        Raises:
            ValueError: if no return numbers stored.
        """
        if "rn" not in self.attributes:
            raise ValueError("Return number attribute not set.")
        I = (self.rn == rn)
        return self.cut(I)

    def get_classes(self):
        """Return the list of unique classes."""
        return self.get_unique_attribute_values("c")

    def get_strips(self):
        # just an alias
        return self.get_pids()

    def get_pids(self):
        """Return the list of unique point source ids"""
        return self.get_unique_attribute_values("pid")

    def get_return_numbers(self):
        """Return the list of unique return numbers (rn_min,...,rn_max)"""
        return self.get_unique_attribute_values("rn")
