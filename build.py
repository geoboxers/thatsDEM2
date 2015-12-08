# -*- coding: utf-8 -*-
"""
SCons wrapper building thatsDEM. Yummy!
@author: simlk
"""
import sys
import os
import subprocess
import tempfile
import urllib2
import zipfile
import md5
import shutil
import logging
import argparse
import glob

LOG = logging.getLogger("build")
HERE = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(HERE, "src")
# Stuff for triangle
TRIANGLE_DIR = os.path.join(SRC_DIR, "triangle", "store")
LIBTRIANGLE = "libtriangle*"
PATCH_TRIANGLE = os.path.join(SRC_DIR, "triangle","triangle_patch.diff")
MD5_TRI="Yjh\xfe\x94o)5\xcd\xff\xb1O\x1e$D\xc4"
URL_TRIANGLE="http://www.netlib.org/voronoi/triangle.zip"

def run(cmd, raise_on_bad_return=True):
    LOG.info("Running %s" % cmd)
    prc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = prc.communicate()
    if prc.returncode != 0 and raise_on_bad_return:
        LOG.warning(str(stderr))
        raise Exception("Unusual return code: %d" % prc.returncode)
    return prc.returncode


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
    LOG.info("Starting patching process of triangle...")
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    LOG.info("Downloading triangle...")
    patching_exception = None
    try:
        with open("triangle.zip", 'wb') as f:
            response = urllib2.urlopen(URL_TRIANGLE)
            assert(response.getcode() == 200)
            f.write(response.read())
        LOG.info("Done...")
        zf = zipfile.ZipFile("triangle.zip")
        zf.extract("triangle.c")
        zf.extract("triangle.h")
        LOG.info("Checking md5 sum of downloaded file...")
        with open("triangle.c", "rb") as f:
            m5 = md5.new(f.read()).digest()
        zf.close()
        assert(m5 == MD5_TRI)
        LOG.info("ok...")
        run("hg init")
        run("hg add triangle.c")
        run("hg commit -m dummy -u dummy")
        run("hg patch " + PATCH_TRIANGLE)
    except Exception as e:
        LOG.exception("Patching process failed!")
        patching_exception = e
    else:
        for src in ("triangle.c", "triangle.h"):
            shutil.copy(src, os.path.join(TRIANGLE_DIR, src))
    finally:
        os.chdir(here)
        try:
            shutil.rmtree(tmpdir)
        except Exception as e:
            print("Failed to delete tmp dir...\n"+str(e))
        if patching_exception:
            raise patching_exception


def build(force_triangle=False, debug=False):
    """Decide if triangle needs to be downloaded and patched."""
    do_triangle = force_triangle
    triangle_libs = glob.glob(os.path.join(TRIANGLE_DIR, LIBTRIANGLE))
    if len(triangle_libs) > 0:
        libtriangle = triangle_libs[0]
        do_triangle |= is_newer(PATCH_TRIANGLE, libtriangle)
    do_triangle |= not os.path.isfile(os.path.join(TRIANGLE_DIR, "triangle.h"))
    if do_triangle:
        patch_triangle()
    LOG.info("Running SCons...")
    rc = subprocess.call("scons do_triangle=%d debug=%d" % (int(do_triangle), int(debug)), shell=True)
    triangle_c = os.path.join(TRIANGLE_DIR, "triangle.c")
    if os.path.isfile(triangle_c):
        # Don't accidently include triangle.c in repo...
        os.remove(triangle_c)
    assert rc == 0

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Build script for thatsDEM. Wraps SCons")
    parser.add_argument("--debug", action="store_true", help="Do a debug build.")
    parser.add_argument("--force_triangle", action="store_true", help="Force downloading and building triangle.")
    pargs = parser.parse_args()
    build(pargs.force_triangle, debug=pargs.debug)
