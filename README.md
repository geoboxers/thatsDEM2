# README #

thatsDEM2!

This project is forked from the [thatsDEM project](https://bitbucket.org/gstudvikler/thatsdem) of the Danish Geodata Agency.
Highly modified - and thus renamed thatsDEM2

### Build instructions ###

Pull the repository and do the following - requires Scons!

```
#!cmd

> python build.py

```
Use --debug for a debug build.
Will require Mingw64 on Windows (setup a proper environment or run from a Mingw64 shell)

### Installation ###
There is no setup.py. You'll need to e.g. modify PYTHONPATH.

### Testing ###
Can be run with nose:

```
#!cmd

> nosetests -v

```
