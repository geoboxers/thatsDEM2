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
"""
Declare names of shared libraries and gather common ctypes definitions.
silyko, June 2016
"""
import os
import sys
import ctypes

LIB_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "lib"))
LIB_TRIPY = os.path.join(LIB_DIR, "libtripy")
LIB_FGEOM = "libfgeom"
LIB_GRID = "libgrid"

if sys.platform.startswith("win"):
    LIB_TRIPY += ".dll"
elif "darwin" in sys.platform:
    LIB_TRIPY += ".dylib"
else:
    LIB_TRIPY += ".so"
# ctypes pointers
LP_CDOUBLE = ctypes.POINTER(ctypes.c_double)
LP_CFLOAT = ctypes.POINTER(ctypes.c_float)
LP_CULONG = ctypes.POINTER(ctypes.c_ulong)
LP_CINT = ctypes.POINTER(ctypes.c_int)
LP_CCHAR = ctypes.POINTER(ctypes.c_char)
