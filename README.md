# Py2VTK
Py2VTK is a Low dependency module for exporting VTK files using Python for visualization/analysis in softwares like Paraview, VisIt or Mayavi.

## Acknowledgments
Py2vtk borrows heavily from PyEvtk as it could be considered a fork. Recreation of its high-level API is planned. Py2vtk also borrows from UVW.

## Current state
Py2vtk is still in development. Planned features include:
* Finalization of a first "release" to make available via Pypi and Spack.
* Recreation of PyEvtk's API
* Easy to use parallel API using mpi4py.
* Support of Dask Arrays