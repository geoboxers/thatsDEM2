import os
import sys
TRIANGLE_DIR = "#src/triangle/store"
IS_WIN = sys.platform.startswith("win")  # So msvc or mingw - use def file
if IS_WIN:
    # Special windows handling - Mingw for now.
    env = Environment(ENV=os.environ, tools=['mingw'])
else:
    env = Environment()

# Use same names on all platforms
env['SHLIBPREFIX'] = "lib"
env['LIBPREFIX']= "lib"

# Check if we requested a debug build...
if int(ARGUMENTS.get("debug", 0)):
    print("DEBUG build...")
    env.Append(CCFLAGS=["-g"])
else:
    env.Append(CCFLAGS=["-O3"])

# Specify librarynames - must match those specified in thatsDEM
env["libtri"] = "triangle"
env["libtripy"] = "tripy"
env["libfgeom"] = "fgeom"
env["libgrid"] = "grid"

if int(ARGUMENTS.get("do_triangle", 0)):
    # Building triangle should be a once off process - but can be forced.
    libtri = env.SConscript(TRIANGLE_DIR+"/SConscript", exports=["env", "IS_WIN"])
    
# Build other libs.
libtripy = env.SConscript("#src/triangle/SConscript", variant_dir="#build/build1",
                          exports=["env", "IS_WIN", "TRIANGLE_DIR"], duplicate=0)
libfgeom = env.SConscript("#src/geometry/SConscript", variant_dir="#build/build2",
                          exports=["env", "IS_WIN"], duplicate=0)
libgrid = env.SConscript("#src/etc/SConscript", variant_dir="#build/build3",
                         exports=["env", "IS_WIN"], duplicate=0)
INSTALL_DIR = "#lib"
env.Install(INSTALL_DIR, libtripy)
env.Install(INSTALL_DIR, libfgeom)
env.Install(INSTALL_DIR, libgrid)






