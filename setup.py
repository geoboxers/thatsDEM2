"""
Simple setup script for thatsDEM2 - use after building with SCons!
"""
from setuptools import setup

setup(name          = 'thatsDEM2',
      version       = '0.1.0',
      description   = 'pointcloud utilities',
      keywords      = 'gis lidar geometry',
      author        = 'Simon Kokkendorff',
      author_email  = 'info@geoboxers.com',
      url   = 'https://github.com/geoboxers/thatsDEM2',
      packages      = ['thatsDEM2'],
      install_requires = ['numpy', 'gdal'],
      package_data ={'thatsDEM2':['lib/lib*']})
