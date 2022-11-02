
*********************
MPI Support in Py2VTK
*********************

Py2VTK has functions to easily create VTK files in a distributed context using ``mpi4py``. Those functions are available under the ``py2vtk.mpi`` submodule.

.. currentmodule:: py2vtk.mpi.api

.. autofunction:: parallelImageToVTK

.. autofunction:: parallelRectilinearGridToVTK

.. autofunction:: parallelStructuredGridToVTK

.. autofunction:: parallelPolyDataToVTK

.. autofunction:: parallelUnstructuredGridToVTK