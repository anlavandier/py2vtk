# Py2VTK

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/anlavandier/py2vtk/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/anlavandier/py2vtk/actions?query=workflow%3Atests)

Py2VTK is a low dependency module for exporting VTK files using Python for visualization/analysis in softwares like Paraview, VisIt or Mayavi.

## Acknowledgments

Py2vtk borrows heavily from [PyEVTK](https://github.com/pyscience-projects/pyevtk) and [UVW](https://github.com/prs513rosewood/uvw).

## Current state

Py2vtk is still in development. Current and future features are listed below in no particular order:

- [ ] Setting up github actions powered CI (**In progress**)
- [ ] Finalization of a first "release" to make available via Pypi and Spack. (**In progress, waiting for CI to be completed**)
- [x] Recreation of PyEvtk's API
- [ ] Parallel API using mpi4py. (**80% Done**)
- [ ] Support of Dask Arrays (**Not started yet**)
- [ ] Add documentation using docstrings and `sphinx` (**Docstrings mostly written, `sphinx` integration not started**)
