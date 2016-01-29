# README #

[![Build Status](https://travis-ci.org/geoboxers/thatsDEM2.svg?branch=master)](https://travis-ci.org/geoboxers/thatsDEM2)

thatsDEM2!

This project is forked from the [thatsDEM project](https://bitbucket.org/gstudvikler/thatsdem) of the Danish Geodata Agency.
Highly modified - and thus renamed thatsDEM2

### Build instructions ###

Pull the repository and do the following - requires Scons!

```

> python build.py

```
Use --debug for a debug build.
Will require Mingw64 on Windows (setup a proper environment or run from a Mingw64 shell).


### Installation ###
There is no setup.py. You'll need to e.g. modify PYTHONPATH.

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
pc = pointcloud.from_las("/data/lidar.laz", attrs=("c","pid","i"))
# It's gonna be a little slow... but works
pc.dump_new_ogr_datasource("db.sqlite", "SQLITE")
# This should be faster
pc.dump_npz("myfile.npz", compressed=True)
# Load back again from the db - with som sql:
pc = pointcloud.from_ogr("db.sqlite", layersql="select * from pointcloud where c=2")
```


