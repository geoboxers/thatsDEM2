# -*- coding: utf-8 -*-
"""
SCons wrapper building thatsDEM. Yummy!
@author: simlk
"""
# Copyright (c) 2015, 2016 Geoboxers <info@geoboxers.com>
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
Perform patching of Jonathan Schewchuk's triangle for 64 bit systems.
Will establish a source directory with patched files and a SConscript for the SCons build.
The SCons build system should then be able to build a static library (libtriangle) and that directory.
"""

import os
import zipfile
import hashlib
import logging
import argparse
import json
import shutil

LOG = logging.getLogger("patch_triangle")
HERE = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(HERE, "src")
# Stuff for triangle
PATCH_TRIANGLE = os.path.join(SRC_DIR, "triangle", "patch.json")
SCONSCRIPT_TRIANGLE = os.path.join(SRC_DIR, "triangle", "build_triangle")
MD5_TRI = "596a68fe946f2935cdffb14f1e2444c4"

def patch_triangle(wrkdir, trizip):
    """Extact triangle source to wrkdir - perfrom patching."""
    LOG.info("Starting patching process of triangle...")
    patching_exception = None
    tri_c_out = os.path.join(wrkdir, "triangle.c")
    try:
        zf = zipfile.ZipFile(trizip)
        zf.extract("triangle.c", wrkdir)
        zf.extract("triangle.h", wrkdir)
        zf.close()
        LOG.info("Checking md5 sum of triangle.c...")
        with open(tri_c_out, "rb") as f:
            tri_bytes = f.read()
            m5 = hashlib.md5(tri_bytes).hexdigest()
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
        if patching_exception:
            raise patching_exception

def main(outdir, trizip):
    # Check that we are not trying to create a subdir to this repository
    if os.path.abspath(outdir).startswith(HERE):
        LOG.exception("You should not use a subdir of this repository!")
        return 1
    if not os.path.isdir(outdir):
        LOG.info("Creating " + outdir)
        os.makedirs(outdir)
    shutil.copy(SCONSCRIPT_TRIANGLE, os.path.join(outdir, "SConscript"))
    patch_triangle(outdir, trizip)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", help="Path to output source dir for patched files and SConscript.")
    parser.add_argument("triangle_zip", help="Path to downloaded triangle zipfile")
    pargs = parser.parse_args()
    main(pargs.outdir, pargs.triangle_zip)
