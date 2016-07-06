# Original work Copyright (c) 2015, Danish Geodata Agency <gst@gst.dk>
# Modified work Copyright (c) 2015 - 2016, Geoboxers <info@geoboxers.com>
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
 Stuff to read / burn vector layers
"""

from osgeo import ogr, osr, gdal
import os
import numpy as np
import time
import logging
LOG = logging.getLogger(__name__)

# placeholder for tile-wkt -
# this token will be replaced by actual wkt in run time.
EXTENT_WKT = "WKT_EXT"


def open(cstr, layername=None, layersql=None, extent=None, sql_dialect=None):
    """
    Common opener of an OGR datasource. Use either layername or layersql.
    The layersql argument takes precedence over a layername arg.
    Will directly modify layersql to make the data provider
    do the filtering by extent if using the WKT_EXT token.
    If the extent (xmin,ymin,xmax,ymax) is in a different coord. sys
    than the layer, the layersql should be intelligent enough to handle
    this, e.g. by using st_transform.
    Returns:
        OGR datasource ,  OGR layer
    """
    ds = ogr.Open(cstr)
    if ds is None:
        raise Exception("Failed to open " + cstr)
    if layersql is not None:  # an sql statement will take precedence
        if extent is not None and EXTENT_WKT in layersql:
            wkt = "'" + extent_to_wkt(extent) + "'"
            layersql = layersql.replace(EXTENT_WKT, wkt)
        # restrict to ASCII encodable chars here -
        # don't know what the datasource
        # is precisely and ogr doesn't like unicode.
        layer = ds.ExecuteSQL(str(layersql), dialect=sql_dialect)
    elif layername is not None:  # then a layername
        layer = ds.GetLayerByName(layername)
    else:  # fallback - shapefiles etc, use first layer
        layer = ds.GetLayer(0)
    assert(layer is not None)

    return ds, layer


def extent_to_wkt(extent):
    """Create a Polygon WKT geometry from extent: (xmin, ymin, xmax, ymax)"""
    wkt = "POLYGON(("
    for dx, dy in ((0, 0), (0, 1), (1, 1), (1, 0)):
        wkt += "{0} {1},".format(str(extent[2 * dx]), str(extent[2 * dy + 1]))
    wkt += "{0} {1}))".format(str(extent[0]), str(extent[1]))
    return wkt


def nptype2gdal(dtype):
    """
    Translate a numpy datatype to a corresponding
    GDAL datatype (similar to mappings internal in GDAL/OGR)
    Arg:
        A numpy datatype
    Returns:
        A GDAL datatype (just a member of an enumeration)
    """
    if dtype == np.float32:
        return gdal.GDT_Float32
    elif dtype == np.float64:
        return gdal.GDT_Float64
    elif dtype == np.int32:
        return gdal.GDT_Int32
    elif dtype == np.bool or dtype == np.uint8:
        return gdal.GDT_Byte
    return gdal.GDT_Float64


def npytype2ogrtype(dtype):
    """Impoverished mapping between numpy dtypes and OGR field types."""
    if issubclass(dtype.type, np.float):
        return ogr.OFTReal
    elif issubclass(dtype.type, np.integer):
        return ogr.OFTInteger
    raise TypeError("dtype cannot be mapped to an OGR type.")


def ogrtype2npytype(ogrtype):
    """Impoverished mapping between OGR field types and numpy dtypes."""
    if ogrtype == ogr.OFTReal:
        return np.float64
    elif ogrtype == ogr.OFTInteger:
        return np.int32
    raise TypeError("OGR type cannot be mapped to a numpy dtype.")


def get_extent(georef, shape):
    """Simple method to compute a grid extent.
    Similar to method in grid.py, but included here
    for self containedness."""
    extent = (
        georef[0],
        georef[3] +
        shape[1] *
        georef[5],
        georef[0] +
        shape[0] *
        georef[1],
        georef[3])  # x1,y1,x2,y2
    return extent

# Example of burning vector layer from a shape file into a raster with
# same georeference as an existing image (assuming that projections match!)
# ds = gdal.Open("myraster.tif")
#
# burn_vector_layer("myfile.shp", ds.GetGeoTransform(),
#                    (ds.RasterYSize,ds.RasterXSize), attr="MyAttr",
#                     dtype=np.int32, osr.SpatialReference(ds.GetProjection))


def burn_vector_layer(cstr, georef, shape, layername=None, layersql=None,
                      attr=None, nd_val=0, dtype=np.bool, all_touched=True,
                      burn3d=False, output_srs=None):
    """
    Burn a vector layer. Will use vector_io.open to fetch the layer.
    Layer can be specified by layersql or layername (else first layer).
    Burn 'mode' defaults to a 'mask', but can be also set by an attr or
    as z-value for 3d geoms.

    When the output_srs differs from the projection of the layer, and
    layersql is used to provide filtering on the dataprovider side, the
    layersql should be intelligent enough to handle the 'extent' input
    which will be the grid extent. E.g. by using st_transform.

    Will (always - for now) set a client side spatial filter
    on the layer corresponding to grid extent.
    Args:
        cstr: OGR connection string
        georef: GDAL style georef, as returned by ds.GetGeoTransform()
        shape: Numpy shape of output raster (nrows,ncols)
        ...
        output_srs: osr.SpatialReference instance - projection of output grid.
    Returns:
        A numpy array of the requested dtype and shape.
    """
    # input a GDAL-style georef
    # If executing fancy sql like selecting buffers etc, be sure to add a
    # where ST_Intersects(geom,TILE_POLY) - otherwise its gonna be slow....
    extent = get_extent(georef, shape)  # this is in the output projection!
    ds, layer = open(cstr, layername, layersql, extent)
    input_srs = layer.GetSpatialRef()
    if input_srs is None or output_srs is None:
        is_same_proj = True
    else:
        is_same_proj = input_srs.IsSame(output_srs)
    # The grid extent in as an ogr geometry (in output_srs)
    geom = ogr.CreateGeometryFromWkt(extent_to_wkt(extent))
    if not is_same_proj:
        # This can only happen if both projections are set
        transform = osr.CoordinateTransformation(output_srs, input_srs)
        geom.Transform(transform)
    # The client side filtering here can speed up things, and is not
    # done by GDAL yet
    # Should have no effect when the filtering is done by the dataprovider
    # using layersql.
    # TODO: remove this, when GDAL implements filtering for rasterize.
    layer.SetSpatialFilter(geom)
    A = just_burn_layer(layer, georef, shape, attr, nd_val, dtype, all_touched,
                        burn3d, output_srs)
    if layersql is not None:
        ds.ReleaseResultSet(layer)
    layer = None
    ds = None
    return A


def rasterize_layer(layer, ds_out, attr=None, burn3d=False, all_touched=True):
    """
    Primitive version of just_burn_layer, where ds_out has already been created.
    """
    options = []
    if all_touched:
        options.append('ALL_TOUCHED=TRUE')
    if attr is not None and burn3d:
        raise ValueError("attr and burn3d keywords incompatible.")
    if attr is not None:
        # we want to burn an attribute - take a different path
        options.append('ATTRIBUTE=%s' % attr)
    elif burn3d:
        options.append('BURN_VALUE_FROM=Z')
    if attr is not None:
        ok = gdal.RasterizeLayer(ds_out, [1], layer, options=options)
    else:
        if burn3d:
            # As explained by Even Rouault:
            # default burn val is 255 if not given.
            # So for burn3d we MUST supply burnval=0
            # and 3d part will be added to that.
            burn_val = 0
        else:
            burn_val = 1
        ok = gdal.RasterizeLayer(
            ds_out, [1], layer,
            burn_values=[burn_val], options=options)
    assert ok == 0


def just_burn_layer(layer, georef, shape, attr=None, nd_val=0,
                    dtype=np.bool, all_touched=True, burn3d=False,
                    output_srs=None):
    """
    Burn a vector layer. See documentation for burn_vector_layer.
    It is the callers responsibility to supply a properly filtered layer,
    e.g. by setting a client side spatial filter, or by using a dataprovider
    side filter, e.g. in layersql.
    Args:
        layer: ogr.Layer
        georef: GDAL style georeference
        shape: numpy output shape (nrows, ncols)
        attr: Attribute to burn
        nd_val: No data / prefill value
        dtype: Output numpy datatype
        all_touched: Option to GDAL - burn all touched cells.
        burn3d: Burn z values for 3d geometries (either this or attr)
        output_srs: osr.SpatialReference instance -
                    will warp input geometries to this projection if layer has spatial reference.
    Returns:
        A numpy array of the requested dtype and shape.
    """
    if burn3d and attr is not None:
        raise ValueError("burn3d and attr can not both be set")
    mem_driver = gdal.GetDriverByName("MEM")
    gdal_type = nptype2gdal(dtype)
    mask_ds = mem_driver.Create("dummy",
                                int(shape[1]), int(shape[0]), 1, gdal_type)
    mask_ds.SetGeoTransform(georef)
    mask = np.ones(shape, dtype=dtype) * nd_val
    mask_ds.GetRasterBand(1).WriteArray(mask)  # write nd_val to output

    if output_srs is not None:
        mask_ds.SetProjection(output_srs.ExportToWkt())
    rasterize_layer(layer, mask_ds, attr, burn3d, all_touched)
    A = mask_ds.ReadAsArray().astype(dtype)
    return A


def get_geometries(cstr, layername=None, layersql=None, extent=None, explode=True, set_filter=True):
    """
    Use vector_io.open to fetch a layer,
    read geometries and explode multi-geometries if explode=True.
    If supplied, and set_filter is True, the extent (xmin,ymin,xmax,ymax)
    should be in the same coordinate system as layer.
    Returns:
        A list of OGR geometries.
    """
    # If executing fancy sql like selecting buffers etc, be sure to add a
    # where ST_Intersects(geom,TILE_POLY) - otherwise it's gonna be slow....
    t1 = time.clock()
    ds, layer = open(cstr, layername, layersql, extent)
    if extent is not None and set_filter:
        # This assumes that the extent provided is in the same coordinate
        # system as the layer!!!
        # Not needed when filtering is taken care of in the layersql.
        layer.SetSpatialFilterRect(*extent)
    nf = layer.GetFeatureCount()
    LOG.info("%d feature(s) in layer %s" % (nf, layer.GetName()))
    geoms = []
    for feature in layer:
        geom = feature.GetGeometryRef().Clone()
        # Handle multigeometries here...
        t = geom.GetGeometryType()
        ng = geom.GetGeometryCount()
        geoms_here = [geom]
        if ng > 1:
            if (t != ogr.wkbPolygon and t != ogr.wkbPolygon25D) and (explode):
                # so must be a multi-geometry - explode it
                geoms_here = [geom.GetGeometryRef(i).Clone() for i in range(ng)]
        geoms.extend(geoms_here)

    if layersql is not None:
        ds.ReleaseResultSet(layer)
    layer = None
    ds = None
    t2 = time.clock()
    LOG.debug("Fetching geoms took %.3f s" % (t2 - t1))
    return geoms


def get_features(cstr, layername=None, layersql=None, extent=None, set_filter=True):
    """
    Use vector_io.open to fetch a layer and read all features.
    If supplied, and set_filter is True,
    the extent (xmin,ymin,xmax,ymax) should be in the same coordinate system
    as layer.
    Returns:
        A list of OGR features.
    """

    ds, layer = open(cstr, layername, layersql, extent)
    if extent is not None and set_filter:
        # This assumes that the extent provided is in the same coordinate
        # system as the layer!!!
        # Not needed when filtering is taken care of in the layersql.
        layer.SetSpatialFilterRect(*extent)
    feats = [f for f in layer]
    if layersql is not None:
        ds.ReleaseResultSet(layer)
    layer = None
    ds = None
    return feats


def polygonize_raster_ds(ds, cstr, layername="polys", fmt="ESRI Shapefile", dst_fieldname="DN"):
    """
    Polygonize a raster datasource.
    Args:
        ds: GDAL raster ds (dtype gdal.GDT_Byte or some other integer type).
        cstr: Connection string to NEW ogr datasource, will be deleted if it exists.
        layername: Name of output polygon layer
        fmt: GDAL driver name.
        dst_fieldname: fieldname to create in output layer.
    Returns:
        ogr_datasource, ogr_layer
    """
    ogr_drv = ogr.GetDriverByName(fmt)
    if ogr_drv is None:
        raise ValueError("No driver named: %s" % fmt)
    if os.path.exists(cstr) and fmt.lower() != "memory":
        # Try to delete datasource
        ogr_drv.DeleteDataSource(cstr)
    ogr_ds = ogr_drv.CreateDataSource(cstr)
    srs_wkt = ds.GetProjection()
    srs = osr.SpatialReference(srs_wkt) if srs_wkt else None
    lyr = ogr_ds.CreateLayer(layername, srs, ogr.wkbPolygon)
    fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
    lyr.CreateField(fd)
    dst_field = 0
    # Ok - so now polygonize that - use the mask as ehem... mask...
    gdal.Polygonize(ds.GetRasterBand(1), ds.GetRasterBand(1), lyr, dst_field)
    return ogr_ds, lyr


def polygonize(M, georef, srs=None, dst_fieldname="DN"):
    """
    Polygonize a mask.
    Args:
        M: a numpy 'mask' array.
        georef: GDAL style georeference of mask.
        srs: osr.SpatialReference instance
    Returns:
        OGR datasource, OGR layer
    """
    # polygonize an input Mask (bool or uint8 -todo, add more types)
    # create a GDAL memory raster
    mem_driver = gdal.GetDriverByName("MEM")
    mask_ds = mem_driver.Create("dummy", int(M.shape[1]), int(M.shape[0]), 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(georef)
    if srs is not None:
        mask_ds.SetProjection(srs.ExportToWkt())
    mask_ds.GetRasterBand(1).WriteArray(M)  # write zeros to output
    ds, lyr = polygonize_raster_ds(mask_ds, "dummy", "polys", "Memory")
    lyr.ResetReading()
    return ds, lyr
