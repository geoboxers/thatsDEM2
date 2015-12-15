# -*- coding: utf-8 -*-
"""
SCons wrapper building thatsDEM. Yummy!
@author: simlk
"""
# Copyright (c) 2015, Geoboxers <info@geoboxers.com>
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


import os
import subprocess
import urllib2
import zipfile
import md5
import logging
import argparse
import glob
import json

LOG = logging.getLogger("build")
HERE = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(HERE, "src")
# Stuff for triangle
TRIANGLE_DIR = os.path.join(SRC_DIR, "triangle", "store")
LIBTRIANGLE = "libtriangle*"
PATCH_TRIANGLE = os.path.join(SRC_DIR, "triangle", "patch.json")
MD5_TRI = "Yjh\xfe\x94o)5\xcd\xff\xb1O\x1e$D\xc4"
URL_TRIANGLE = "http://www.netlib.org/voronoi/triangle.zip"


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


def patch_triangle(wrkdir):
    LOG.info("Starting patching process of triangle...")
    LOG.info("Downloading triangle...")
    patching_exception = None
    trizip = os.path.join(wrkdir, "triangle.zip")
    tri_c_out = os.path.join(wrkdir, "triangle.c")
    try:
        with open(trizip, "wb") as f:
            response = urllib2.urlopen(URL_TRIANGLE)
            assert(response.getcode() == 200)
            f.write(response.read())
        LOG.info("Done...")
        zf = zipfile.ZipFile(trizip)
        zf.extract("triangle.c", wrkdir)
        zf.extract("triangle.h", wrkdir)
        zf.close()
        LOG.info("Checking md5 sum of downloaded file...")
        with open(tri_c_out, "rb") as f:
            tri_bytes = f.read()
            m5 = md5.new(tri_bytes).digest()
        assert(m5 == MD5_TRI)
        LOG.info("ok...")
        # A hassle to fiddle with git apply, hg pacth etc...
        # This workz!!
        with open(PATCH_TRIANGLE) as f:
            patch = json.load(f)
        if "global" in patch:
            for fr, to in patch["global"]:
                LOG.debug("Replacing %s with %s" % (fr, to))
                tri_bytes = tri_bytes.replace(fr, to)
        lines = tri_bytes.splitlines()
        if "local" in patch:
            LOG.debug("Replacing in lines..")
            for local in patch["local"]:
                for line in local["lines"]:
                    assert local["from"] in lines[line]
                    lines[line] = lines[line].replace(local["from"], local["to"])
        if "insert_after" in patch:
            # For this to work - insertions MUST be sorted afte line numbers
            LOG.debug("Inserting...")
            lines_out = []
            cl = 0
            for insert in patch["insert_after"]:
                ln = insert["line"] + 1  # inserting AFTER
                assert ln > cl
                LOG.debug("cl: %d, ln: %d" % (cl, ln))
                lines_out.extend(lines[cl:ln])
                lines_out.extend(insert["lines"])
                cl = ln
            lines_out.extend(lines[cl:])
        else:
            lines_out = lines

    except Exception as e:
        LOG.exception("Patching process failed!")
        patching_exception = e
    else:
        with open(tri_c_out, "wb") as f:
            f.write("\n".join(lines_out))
    finally:
        if os.path.isfile(trizip):
            os.remove(trizip)
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
        patch_triangle(TRIANGLE_DIR)
    LOG.info("Running SCons...")
    rc = subprocess.call("scons do_triangle=%d debug=%d" % (int(do_triangle), int(debug)), shell=True)
    triangle_c = os.path.join(TRIANGLE_DIR, "triangle.c")
    if os.path.isfile(triangle_c):
        # Don't accidently include triangle.c in repo...
        os.remove(triangle_c)
    assert rc == 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Build script for thatsDEM. Wraps SCons")
    parser.add_argument("--debug", action="store_true", help="Do a debug build.")
    parser.add_argument("--force_triangle", action="store_true", help="Force downloading and building triangle.")
    pargs = parser.parse_args()
    build(pargs.force_triangle, debug=pargs.debug)
