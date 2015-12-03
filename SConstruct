import os
import sys
import subprocess
import tempfile
import urllib2
import zipfile
import md5
import shutil


SRC_DIR = Dir("#src").abspath
IGNORE_TRIANGLE_DIR = os.path.join(SRC_DIR, "triangle", "ignore")
PATCH_TRIANGLE = os.path.join(SRC_DIR, "triangle","triangle_patch.diff")
MD5_TRI="Yjh\xfe\x94o)5\xcd\xff\xb1O\x1e$D\xc4"
URL_TRIANGLE="http://www.netlib.org/voronoi/triangle.zip"
IS_WIN = sys.platform.startswith("win")  # So msvc or mingw - use def file

if not os.path.isdir(IGNORE_TRIANGLE_DIR):
    os.mkdir(IGNORE_TRIANGLE_DIR)

if IS_WIN:
    # Special windows handling - Mingw for now.
    env = Environment(SHLIBPREFIX="lib", ENV=os.environ, tools=['mingw'])
else:
    env = Environment(SHLIBPREFIX="lib")

# Check if we requested a debug build...
if ARGUMENTS.get("debug"):
    env.Append(CCFLAGS=["-g"])
else:
    env.Append(CCFLAGS=["-O2"])

# Specify librarynames - must match those specified in thatsDEM
env["libtri"] = "triangle"
env["libtripy"] = "tripy"
env["libfgeom"] = "fgeom"
env["libgrid"] = "grid"

def is_newer(p1, p2):
    """We need to hack scons a bit, since we wanna download and patch
    triangle. This is used to determine if a target is newer than source,
    and to determine if the patching process is needed.
    """
    if not os.path.exists(p1):
        return False
    if not os.path.exists(p2):
        return True
    return os.path.getmtime(p1) > os.path.getmtime(p2)

def patch_triangle():
    here = os.getcwd()
    print("Starting patching process of triangle...")
    tmpdir=tempfile.mkdtemp()
    os.chdir(tmpdir)
    print("Downloading triangle...")
    patching_exception = None
    try:
        with open("triangle.zip", 'wb') as f:
            response = urllib2.urlopen(URL_TRIANGLE)
            assert(response.getcode()==200)
            f.write(response.read())
        print("Done...")
        zf=zipfile.ZipFile("triangle.zip")
        zf.extract("triangle.c")
        zf.extract("triangle.h")
        print("Checking md5 sum of downloaded file...")
        with open("triangle.c","rb") as f:
            m5=md5.new(f.read()).digest()
        zf.close()
        assert(m5==MD5_TRI)
        print("ok...")
        rc = subprocess.call("hg init", shell=True)
        assert(rc == 0)
        rc = subprocess.call("hg add triangle.c", shell=True)
        assert(rc == 0)
        rc = subprocess.call("hg commit -m dummy -u dummy", shell=True)
        assert(rc == 0)
        rc = subprocess.call("hg patch " + PATCH_TRIANGLE, shell=True)
        assert(rc == 0)
    
    except Exception as e:
        print("Patching process failed!")
        print(str(e))
        patching_exception = e
    else:
        for src in ("triangle.c", "triangle.h"):
            shutil.copy(src, os.path.join(IGNORE_TRIANGLE_DIR, src))
    finally:
        os.chdir(here)
        try:
            shutil.rmtree(tmpdir)
        except Exception as e:
            print("Failed to delete tmp dir...\n"+str(e))
        if patching_exception:
            raise patching_exception
   
# Check if triangle is patched
if (not env.GetOption('clean')) and (not os.path.isfile(os.path.join(IGNORE_TRIANGLE_DIR, "triangle.c"))):
    patch_triangle()

# Build triangle, etc.
libtri, libtripy = env.SConscript("#src/triangle/SConscript", variant_dir="#build/build1", exports=["env", "IS_WIN"], duplicate=0)
libfgeom = env.SConscript("#src/geometry/SConscript", variant_dir="#build/build2", exports=["env", "IS_WIN"], duplicate=0)
libgrid = env.SConscript("#src/etc/SConscript", variant_dir="#build/build3", exports=["env", "IS_WIN"], duplicate=0)
INSTALL_DIR = "#lib"
env.Install(INSTALL_DIR, libtri)
env.Install(INSTALL_DIR, libtripy)
env.Install(INSTALL_DIR, libfgeom)
env.Install(INSTALL_DIR, libgrid)






