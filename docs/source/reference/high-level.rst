.. _hl:

********************
High-level Functions
********************

Py2VTK natively implements `PyEVTK's API <https://github.com/pyscience-projects/pyevtk>`_
with some minor changes in the order of keyword arguments and some new arguments,
most notably the possibility to decide which format to use when writing the file to disk.
The options are:

* "raw":
    Write the data in bytes. This is space efficient and supported by VTK but means
    that the ensuing XML files are not valid.
* "ascii":
    Write the data in readable ascii characters. This is very inefficient and
    should only be used for debugging.
* "binary":
    Write the data in base64 with the possibility of compressing the data.

When using the latter format, it is possible to compress the data using ``zlib`` or ``lzma``.

Functions writing serial VTK XML files
-------------------------------------------
.. currentmodule:: py2vtk.api.serial

.. autofunction:: imageToVTK

.. autofunction:: gridToVTK

.. autofunction:: polyDataToVTK

.. autofunction:: pointsToVTK

.. autofunction:: linesToVTK

.. autofunction:: unstructuredGridToVTK

.. autofunction:: cylinderToVTK

Functions writing parallel VTK XML files
-------------------------------------------

.. note::
    Despite being called parllel VTK files, this file format is expected to
    be written in serial. As such, the following functions are serial.

.. currentmodule:: py2vtk.api.parallel

.. autofunction:: writeParallelVTKImageData

.. autofunction:: writeParallelVTKGrid

.. autofunction:: writeParallelVTKPolyData

.. autofunction:: writeParallelVTKUnstructuredGrid