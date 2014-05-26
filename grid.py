######################################
##  Grid class  - just a numpy array and some metadata + some usefull methods             
####################################
import numpy as np
import os
from osgeo import gdal
import ctypes
LIBDIR=os.path.realpath(os.path.join(os.path.dirname(__file__),"../lib"))
LIBNAME="libgrid"
XY_TYPE=np.ctypeslib.ndpointer(dtype=np.float64,flags=['C','O','A','W'])
GRID_TYPE=np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags=['C','O','A','W'])
Z_TYPE=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags=['C','O','A','W'])
LP_CDOUBLE=ctypes.POINTER(ctypes.c_double)
GEO_REF_ARRAY=ctypes.c_double*4
lib=np.ctypeslib.load_library(LIBNAME, LIBDIR)
#void wrap_bilin(double *grid, double *xy, double *out, double *geo_ref, double nd_val, int nrows, int ncols, int npoints)
lib.wrap_bilin.argtypes=[GRID_TYPE,XY_TYPE,Z_TYPE,LP_CDOUBLE,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int]
lib.wrap_bilin.restype=None
#If there's no natural nodata value connected to the grid, it is up to the user to supply a nd_val which is not a regular grid value.
#If supplied geo_ref should be a 'sequence' of len 4 (duck typing here...)


def fromGDAL(path,upcast=False):
	ds=gdal.Open(path)
	a=ds.ReadAsArray()
	if upcast:
		a=a.astype(np.float64)
	geo_ref=ds.GetGeoTransform()
	nd_val=ds.GetRasterBand(1).GetNoDataValue()
	ds=None
	return Grid(a,geo_ref,nd_val)

def bilinear_interpolation(grid,xy,nd_val,geo_ref=None):
	if geo_ref is not None:
		if len(geo_ref)!=4:
			raise Exception("Geo reference should be sequence of len 4, xulcenter, cx, yulcenter, cy")
		geo_ref=GEO_REF_ARRAY(*geo_ref)
	p_geo_ref=ctypes.cast(geo_ref,LP_CDOUBLE)  #null or pointer to geo_ref
	grid=np.require(grid,dtype=np.float64,requirements=['A', 'O', 'C','W'])
	xy=np.require(xy,dtype=np.float64,requirements=['A', 'O', 'C','W'])
	out=np.zeros((xy.shape[0],),dtype=np.float64)
	lib.wrap_bilin(grid,xy,out,p_geo_ref,nd_val,grid.shape[0],grid.shape[1],xy.shape[0])
	return out

class Grid(object):
	"""
	Grid abstraction class.
	Contains a numpy array and metadata like geo reference.
	"""
	def __init__(self,arr,geo_ref,nd_val=None):
		self.grid=arr
		self.geo_ref=geo_ref
		self.nd_val=nd_val
		#and then define some useful methods...
	def interpolate(self,xy,nd_val=None):
		#If the grid does not have a nd_val, the user must supply one here...
		if self.nd_val is None:
			if nd_val is None:
				raise Exception("No data value not supplied...")
		else:
			if nd_val is not None:
				raise Warning("User supplied nd-val not used as grid already have one...")
			nd_val=self.nd_val
		cx=self.geo_ref[1]
		cy=self.geo_ref[5]
		cell_georef=[self.geo_ref[0]+0.5*cx,cx,self.geo_ref[3]+0.5*cy,-cy]  #geo_ref used in interpolation ('corner' coordinates...)
		return bilinear_interpolation(self.grid,xy,nd_val,cell_georef)
	def save(self,fname,format="GTiff"):
		#TODO: map numpy types to gdal types better - done internally in gdal I think...
		if self.grid.dtype==np.float32:
			dtype=gdal.GDT_Float32
		elif self.grid.dtype==np.float64:
			dtype=gdal.GDT_Float64
		elif self.grid.dtype==np.int32:
			dtype=gdal.GDT_Int32
		elif self.grid.dtype==np.bool:
			dtype=gdal.GDT_Byte
		else:
			return False #TODO....
		driver=gdal.GetDriverByName(format)
		if driver is None:
			return False
		if os.path.exists(fname):
			try:
				driver.Delete(fname)
			except Exception, msg:
				print msg
			else:
				print("Overwriting %s..." %fname)	
		else:
			print("Saving %s..."%fname)
		dst_ds=driver.Create(fname,self.grid.shape[1],self.grid.shape[0],1,dtype)
		dst_ds.SetGeoTransform(self.geo_ref)
		band=dst_ds.GetRasterBand(1)
		if self.nd_val is not None:
			band.SetNoDataValue(self.nd_val)
		band.WriteArray(self.grid)
		dst_ds=None
		return True
	def get_bounds(self):
		x1=self.geo_ref[0]
		y2=self.geo_ref[3]
		x2=x1+self.grid.shape[1]*self.geo_ref[1]
		y1=y2+self.grid.shape[0]*self.geo_ref[5]
		return (x1,y1,x2,y2)
	def correlate(self,other):
		pass #TODO
	def get_hillshade(self,light=(1,-1,-4),sigma=0,remove_extreme=False):
		#THIS is just stupid - should be done in c....
		print("Casting shadow...")
		light=np.array(light)
		light=light/(np.sqrt((light**2).sum()))
		print("Light: %s" %repr(light))
		M=(self.grid==self.nd_val)
		dx=np.zeros_like(self.grid)
		dy=np.zeros_like(self.grid)
		dx[:,0:self.grid.shape[1]-1]=self.grid[:,1:]-self.grid[:,0:self.grid.shape[1]-1]
		dy[0:self.grid.shape[0]-1]=self.grid[0:self.grid.shape[0]-1,:]-self.grid[1:,:]
		if remove_extreme and M.any(): #fast and dirty - but only works when nd-value is large compared to data!!!!!
			print("Deleting extreme slopes (probably from no-data)")
			dx[np.fabs(dx)>100]=0
			dy[np.fabs(dy)>100]=0
		if sigma>0 and False: #TODO
			dx=image.filters.gaussian_filter(dx,sigma)
			dy=image.filters.gaussian_filter(dy,sigma)
		X=np.sqrt(dx**2+dy**2+1)
		return Grid((dx*light[0]/X-dy*light[1]/X-light[2]/X)/np.sqrt(3),self.geo_ref) #cast shadow