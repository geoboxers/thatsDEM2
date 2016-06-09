# README #

[![Build Status](https://travis-ci.org/geoboxers/thatsDEM2.svg?branch=master)](https://travis-ci.org/geoboxers/thatsDEM2)

thatsDEM2!

This project is forked from the [thatsDEM project](https://bitbucket.org/gstudvikler/thatsdem) of the Danish Geodata Agency.
Highly modified - and thus renamed thatsDEM2

### Build instructions ###

Pull the repository and do the following - requires Scons!
Will require Mingw64 on Windows (setup a proper environment or run from a Mingw64 shell).

```

> scons

```
Use debug=1 for a debug build.
The triangulation engine can use triangle by Jonathan Richard Schewchuk.
In order to enable that, you'll need to download triangle.zip from http://www.netlib.org/voronoi/
and patch the source (for 64-bit systems) using the included utility, patch_triangle.py:

```

> python patch_triangle.py <out_dir> <path_to_triangle.zip>

```
and then, specify:

```

> scons with-triangle=<out_dir>

```

If NOT using triangle, the triangulation engine will fall back to the slower and less robust scipy.spatial.Delaunay
 

### Installation ###
After building with SCons you can use the simple setup script to install:

```

> python setup.py install

```
Or you can simply copy the thatsDEM2 subfolder to somewhere on your PYTHONPATH.

### Testing ###
Can be run with nose:

```

> nosetests -v

```

### Example ###

Have you ever wanted to convert a laz file to a sqlite db?
```python
from thatsDEM2 import pointcloud
# requires laspy or slash
pc = pointcloud.LidarPointcloud.from_las("/data/lidar.laz", attrs=("c","pid","i"))
# It's gonna be a little slow... but works
pc.dump_new_ogr_datasource("db.sqlite", "SQLITE")
# This should be faster
pc.dump_npz("myfile.npz", compressed=True)
# Load back again from the db - with som sql:
pc = pointcloud.LidarPointcloud.from_ogr("db.sqlite", layersql="select * from pointcloud where c=2")
```

### Another example ###
```python
pc1.triangulate() # will use triangle
g1 = pc1.get_grid(ncols=1000, nrows=1000, method="triangulation")
pc2 = pc1.copy()
pc2.sort_spatially(2)
g2 = pc2.get_grid(ncols=1000, nrows=1000, method="idw_filter", srad=2)
h2 = g2.get_hillshade()
h2.save("hillshade.tif", dco=["TILED=YES", "COMPRESS=LZW"])
```
