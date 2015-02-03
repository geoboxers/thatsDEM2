# Copyright (c) 2015, Danish Geodata Agency <gst@gst.dk>
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
#########################
## IO library functions...
#########################

from osgeo import ogr
import time


#A lazy, hacky global layer, that can be set and will override subsequen args to get_geometries
GLOBAL_LAYER=None

def set_global_layer(layer):
	global GLOBAL_LAYER
	GLOBAL_LAYER=layer

def get_geometries(cstr, layername=None, layersql=None, extent=None):
	#very simplistic: if layername is none, we assume we want the first layer in datasource (true for shapefiles, etc).
	t1=time.clock()
	if GLOBAL_LAYER is not None:
		layer=GLOBAL_LAYER
		is_global=True
	else:
		is_global=False
		ds=ogr.Open(cstr)
		if ds is None:
			raise Exception("Failed to open "+cstr)
		if layersql is not None: #an sql statement will take precedence
			layer=ds.ExecuteSQL(layersql)
		elif layername is not None:  #then a layername
			layer=ds.GetLayerByName(layername)
		else: #fallback - shapefiles etc, use first layer
			layer=ds.GetLayer(0)
		assert(layer is not None)
	if extent is not None:
		layer.SetSpatialFilterRect(*extent)
	else:
		layer.SetSpatialFilterRect(None) #reset
	nf=layer.GetFeatureCount()
	print("%d feature(s) in layer %s" %(nf,layer.GetName()))
	geoms=[]
	for i in xrange(nf):
		feature=layer.GetNextFeature()
		geom=feature.GetGeometryRef().Clone()
		#Handle multigeometries here...
		t=geom.GetGeometryType()
		ng=geom.GetGeometryCount()
		geoms_here=[geom]
		if ng>1:
			if t!=ogr.wkbPolygon and t!=ogr.wkbPolygon25D:
				#so must be a multi-geometry
				geoms_here=[geom.GetGeometryRef(i).Clone() for i in range(ng)]
		geoms.extend(geoms_here)
	if not is_global:
		if layersql is not None:
			ds.ReleaseResultSet(layer)
		layer=None
		ds=None
	t2=time.clock()
	print("Fetching geoms took %.3f s" %(t2-t1))
	return geoms

def read(path,attrs=[]):
	#not used at the moment... TODO: reimplement
	ds=ogr.Open(path)
	if ds is None:
		return []
	layer=ds.GetLayer(0)
	nf=layer.GetFeatureCount()
	feats=[]
	print("%d feature(s) in %s" %(nf,path))
	for i in xrange(nf):
		feature=layer.GetNextFeature()
		feats.appen(feature)
	ds=None
	return feats
	