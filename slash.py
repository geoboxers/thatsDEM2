import sys,os,ctypes,platform
import numpy as np
LIBDIR=os.path.realpath(os.path.join(os.path.dirname(__file__),"../lib"))
LIBNAME="slash"
#'64' not appended to libname anymore
if sys.platform.startswith("win"):
	LIBNAME+=".dll"
	os.environ["PATH"]+=";"+LIBDIR
elif "darwin" in sys.platform:
	LIBNAME+=".dylib"
else:
	LIBNAME+=".so"

LP_CDOUBLE=ctypes.POINTER(ctypes.c_double)
LP_CINT=ctypes.POINTER(ctypes.c_int)
#lib_name=os.path.join(os.path.dirname(__file__),LIBNAME)
lib_name=os.path.join(LIBDIR,LIBNAME)
#print("Loading %s" %lib_name)
lib=ctypes.cdll.LoadLibrary(lib_name)
lib.las_open.argtypes=[ctypes.c_char_p,ctypes.c_char_p]
lib.las_open.restype=ctypes.c_void_p
lib.las_close.argtypes=[ctypes.c_void_p]
lib.las_close.restype=None
lib.py_get_num_records.argtypes=[ctypes.c_void_p]
lib.py_get_num_records.restype=ctypes.c_ulong
#unsigned long py_set_mask(LAS *h, char mask, int *cs, double *xy_box, double *z_box, int nc)
lib.py_set_mask.argtypes=[ctypes.c_void_p, ctypes.c_char_p, LP_CINT, LP_CDOUBLE, LP_CDOUBLE, ctypes.c_int]
lib.py_set_mask.restype=ctypes.c_ulong
#unsigned long py_get_records(LAS *h, double *xy, double *z, int *c, int *pid, int *return_number, char *mask, unsigned long buf_size)
lib.py_get_records.argtypes=[ctypes.c_void_p,LP_CDOUBLE,LP_CDOUBLE,LP_CINT,LP_CINT,LP_CINT, ctypes.c_char_p, ctypes.c_ulong]
lib.py_get_records.restype=ctypes.c_ulong


		
class LasFile(object):
	def __init__(self,path):
		self.plas=lib.las_open(path,"rb")
		if self.plas is None:
			raise ValueError("Failed to open input file...")
		self.n_records=lib.py_get_num_records(self.plas)
		self.path=path
		self.mask=None
		self.mask_count=None
	def close(self):
		if self.plas is not None:
			lib.las_close(self.plas)
			self.plas=None
	def get_number_of_records(self):
		return self.n_records
	def set_mask(self,xy_box=None,z_box=None,cs=None):
		self.mask=np.zeros((self.n_records,),dtype=np.bool)
		if xy_box is not None:
			xy_box=np.require(np.asarray(xy_box,dtype=np.float64),requirements=['A','O','C'])
			assert( xy_box.size==4)
			p_xy_box=xy_box.ctypes.data_as(LP_CDOUBLE)
		else:
			p_xy_box=None
		if z_box is not None:
			z_box=np.require(np.asarray(z_box,dtype=np.float64),requirements=['A','O','C'])
			assert( z_box.size==2)
			p_z_box=z_box.ctypes.data_as(LP_CDOUBLE)
		else:
			p_z_box=None
		if cs is not None and len(cs)>0:
			cs=np.require(np.asarray(cs,dtype=np.int32),requirements=['A','O','C'])
			p_cs=cs.ctypes.data_as(LP_CINT)
			ncs=cs.size
		else:
			p_cs=None
			ncs=0
		p_mask=self.mask.ctypes.data_as(ctypes.c_char_p)
		self.mask_count=lib.py_set_mask(self.plas, p_mask,p_cs, p_xy_box, p_z_box, ncs)
		
	def read_records(self,return_z=True,return_c=True,return_pid=True,return_ret_number=False):
		#Hmmm for now - always read from beginning of file - to match mask values... otherwise we would need to keep track of current index...
		#read everything or just whats in mask...
		if (self.mask is None):
			n=self.n_records
			p_mask=None
		else:
			n=self.mask_count
			p_mask=self.mask.ctypes.data_as(ctypes.c_char_p)
			
		xy=np.empty((n,2),dtype=np.float64)
		p_xy=xy.ctypes.data_as(LP_CDOUBLE)
		ret=dict()
		ret["xy"]=xy
		if return_z:
			z=np.empty((n,),dtype=np.float64)
			p_z=z.ctypes.data_as(LP_CDOUBLE)
			ret["z"]=z
		else:
			p_z=None
			ret["z"]=None
		if return_c:
			c=np.empty((n,),dtype=np.int32)
			p_c=c.ctypes.data_as(LP_CINT)
			ret["c"]=c
		else:
			p_c=None
			ret["c"]=None
		if return_pid:
			pid=np.empty((n,),dtype=np.int32)
			p_pid=pid.ctypes.data_as(LP_CINT)
			ret["pid"]=pid
		else:
			p_pid=None
			ret["pid"]=None
		if return_ret_number:
			rn=np.empty((n,),dtype=np.int32)
			p_rn=rn.ctypes.data_as(LP_CINT)
			ret["rn"]=rn
		else:
			p_rn=None
			ret["rn"]=None
		if n>0:
			n_read=lib.py_get_records(self.plas, p_xy, p_z, p_c, p_pid, p_rn, p_mask, n)
			assert(n_read==n)
		return ret


if __name__=="__main__":
	lasf=LasFile(sys.argv[1])
	print("%d points in %s" %(lasf.get_number_of_records(),sys.argv[1]))
	ret=lasf.read_records()
	xy=ret["xy"]
	z=ret["z"]
	c=ret["c"]
	pid=ret["pid"]
	x1,y1=xy.min(axis=0)
	x2,y2=xy.max(axis=0)
	z1=z.min()
	z2=z.max()
	print("XY: %.2f %.2f %.2f %.2f, Z: %.2f %.2f %.2f" %(x1,y1,x2,y2,z1,z2,z.mean()))
	print("classes: %s" %np.unique(c))
	print("point ids: %s" %np.unique(pid))
	
