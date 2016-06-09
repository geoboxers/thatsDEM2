# Copyright (c) 2016, Geoboxers <info@geoboxers.com>
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
import os
import sys
IS_WIN = sys.platform.startswith("win")  # So msvc or mingw - use def file
if IS_WIN:
    # Special windows handling - Mingw for now.
    env = Environment(ENV=os.environ, tools=['mingw'])
else:
    env = Environment()

# Use same names on all platforms
env['SHLIBPREFIX'] = "lib"
env['LIBPREFIX'] = "lib"

# Check if we requested a debug build...
if int(ARGUMENTS.get("debug", 0)):
    print("DEBUG build...")
    env.Append(CCFLAGS=["-g"])
else:
    env.Append(CCFLAGS=["-O3"])

env.Append(CCFLAGS=["-Wall", "-pedantic"])
# Append a define to fix exporting functions in msvc
env.Append(CPPDEFINES=["_EXPORTING"])
# Specify librarynames - must match those specified in thatsDEM
env["libtri"] = "triangle"
env["libtripy"] = "tripy"
env["libfgeom"] = "fgeom"
env["libgrid"] = "grid"

TRIANGLE_DIR = ARGUMENTS.get("with-triangle", None)

if TRIANGLE_DIR:
    # Building pathced version of triangle - the user should know what she's doing!
    print("Building against triangle! See src/triangle/README")
    TRIANGLE_DIR = os.path.abspath(TRIANGLE_DIR)
    env.SConscript(TRIANGLE_DIR + "/SConscript", exports=["env", "IS_WIN"])
else:
    print("with-triangle=<triangle_dir> not specified. Won't use triangle!")
# Build other libs.
libtripy = env.SConscript("#src/triangle/SConscript", variant_dir="#build/build1",
                          exports=["env", "IS_WIN", "TRIANGLE_DIR"], duplicate=0)
libfgeom = env.SConscript("#src/geometry/SConscript", variant_dir="#build/build2",
                          exports=["env", "IS_WIN"], duplicate=0)
libgrid = env.SConscript("#src/etc/SConscript", variant_dir="#build/build3",
                         exports=["env", "IS_WIN"], duplicate=0)
INSTALL_DIR = "#thatsDEM2/lib"  # Perhaps _import_ from thatsDEM2 ?
env.Install(INSTALL_DIR, libtripy)
env.Install(INSTALL_DIR, libfgeom)
env.Install(INSTALL_DIR, libgrid)
