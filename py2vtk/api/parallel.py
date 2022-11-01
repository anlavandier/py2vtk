""" High level API for parallel VTK files"""
import numpy as np

from ..core.vtkfiles import (
    VtkParallelFile,
    VtkPImageData,
    VtkPPolyData,
    VtkPRectilinearGrid,
    VtkPStructuredGrid,
    VtkPUnstructuredGrid,
)
from ..utilities.utils import _addDataToParallelFile, _addFieldDataToParallelFile

__all__ = [
    "writeParallelVTKImageData",
    "writeParallelVTKGrid",
    "writeParallelVTKPolyData",
    "writeParallelVTKUnstructuredGrid",
]


# ==============================================================================
def writeParallelVTKImageData(
    path,
    starts,
    ends,
    sources,
    dimension=None,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    ghostlevel=0,
    cellData=None,
    pointData=None,
    fieldData=None,
    format="binary",
):
    """
    Writes a parallel vtk file from grid-like data:
    VTKStructuredGrid or VTKRectilinearGrid

    Parameters
    ----------
    path : str
        name of the file without extension.

    starts : list
        list of 3-tuple representing where each source file starts
        in each dimension.

    ends : list
        list of 3-tuple representing where each source file ends
        in each dimension

    dimension : tuple or None, optional
        dimension of the image.

    source : list
        list of the relative paths of the source files where the actual data is found.

    origin : tuple, optional
        grid origin.
        The default is (0.0, 0.0, 0.0).

    spacing : tuple, optional
        grid spacing.
        The default is (1.0, 1.0, 1.0).

    ghostlevel : int, default=0
        Number of cells which are shared between neighbouring files.

    pointData : dict
        dictionnary containing the information about the arrays
        containing node centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)

    cellData :
        dictionnary containing the information about the arrays
        containing cell centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
    """
    # Check that every source as a start and an end
    assert len(starts) == len(ends) == len(sources)

    # Get the extension + check that it's consistent accros all source files
    common_ext = sources[0].split(".")[-1]
    assert all(
        s.split(".")[-1] == common_ext for s in sources
    ), "All sources need to share the same extension"
    if common_ext != "vti":
        raise ValueError(
            f"Sources must be VTKImageData ('.vti') and not {common_ext} files"
        )

    w = VtkParallelFile(path, VtkPImageData, format=format)
    start = (0, 0, 0)

    if dimension is None:
        dimension = tuple(np.max(np.array(ends), axis=0))

    w.openGrid(
        start=start,
        end=dimension,
        origin=origin,
        spacing=spacing,
        ghostlevel=ghostlevel,
    )

    _addDataToParallelFile(w, cellData=cellData, pointData=pointData)

    for start_source, end_source, source in zip(starts, ends, sources):
        w.addPiece(source, start_source, end_source)

    _addFieldDataToParallelFile(w, fieldData=fieldData)

    w.closeGrid()
    w.save()

    return w.getFileName()


# ==============================================================================
def writeParallelVTKGrid(
    path,
    coordsDtype,
    starts,
    ends,
    sources,
    dimension=None,
    ghostlevel=0,
    cellData=None,
    pointData=None,
    fieldData=None,
    format="binary",
):
    """
    Writes a parallel vtk file from grid-like data:
    VTKStructuredGrid or VTKRectilinearGrid

    Parameters
    ----------
    path : str
        name of the file without extension.

    coordsDtype : numpy.dtype
        the dtype of the coordinates.

    starts : list
        list of 3-tuple representing where each source file starts
        in each dimension.

    ends : list
        list of 3-tuple representing where each source file ends
        in each dimension

    source : list
        list of the relative paths of the source files where the actual data is found.

    ghostlevel : int, default=0
        Number of cells which are shared between neighbouring files.

    pointData : dict
        dictionnary containing the information about the arrays
        containing node centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)

    cellData :
        dictionnary containing the information about the arrays
        containing cell centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
    """
    # Check that every source as a start and an end
    assert len(starts) == len(ends) == len(sources)

    # Get the extension + check that it's consistent accros all source files
    common_ext = sources[0].split(".")[-1]
    assert all(s.split(".")[-1] == common_ext for s in sources)

    if common_ext == "vts":
        ftype = VtkPStructuredGrid
        is_Rect = False
    elif common_ext == "vtr":
        ftype = VtkPRectilinearGrid
        is_Rect = True
    else:
        raise ValueError(
            "This functions is meant to work only with VTK Structured grids and VTK Rectilinear grids"
        )

    w = VtkParallelFile(path, ftype, format=format)
    start = (0, 0, 0)
    dtype = coordsDtype
    if dimension is None:
        dimension = tuple(np.max(np.array(ends), axis=0))

    w.openGrid(start=start, end=dimension, ghostlevel=ghostlevel)

    _addDataToParallelFile(w, cellData=cellData, pointData=pointData)

    if is_Rect:
        w.openElement("PCoordinates")
        w.addPData("x_coordinates", dtype=dtype, ncomp=1)
        w.addPData("y_coordinates", dtype=dtype, ncomp=1)
        w.addPData("z_coordinates", dtype=dtype, ncomp=1)
        w.closeElement("PCoordinates")
    else:
        w.openElement("PPoints")
        w.addPData("points", dtype=dtype, ncomp=3)
        w.closeElement("PPoints")

    for start_source, end_source, source in zip(starts, ends, sources):
        w.addPiece(source, start_source, end_source)

    _addFieldDataToParallelFile(w, fieldData=fieldData)

    w.closeGrid()
    w.save()
    return w.getFileName()


# ==============================================================================
def writeParallelVTKPolyData(
    path,
    coordsDtype,
    sources,
    ghostlevel=0,
    cellData=None,
    pointData=None,
    fieldData=None,
    format="binary",
):
    """
    Writes a parallel VTK PolyData.

    Parameters
    ----------
    path : str
        name of the file without extension.

    coordsDtype : numpy.dtype
        the dtype of the coordinates.

    source : list
        list of the relative paths of the source files where the actual data is found.

    ghostlevel : int, default=0
        Number of cells which are shared between neighbouring files.

    pointData : dict
        dictionnary containing the information about the arrays
        containing node centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)

    cellData :
        dictionnary containing the information about the arrays
        containing cell centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
    """
    # Get the extension + check that it's consistent accros all source files
    common_ext = sources[0].split(".")[-1]
    assert all(s.split(".")[-1] == common_ext for s in sources)

    if common_ext != "vtp":
        raise ValueError(
            f"Sources must be VTKPolyData ('.vtp') and not {common_ext} files"
        )

    w = VtkParallelFile(path, VtkPPolyData, format=format)

    w.openGrid(ghostlevel=ghostlevel)

    w.openElement("PPoints")
    w.addPData("points", dtype=coordsDtype, ncomp=3)
    w.closeElement("PPoints")
    _addDataToParallelFile(w, cellData=cellData, pointData=pointData)

    for source in sources:
        w.addPiece(source)

    _addFieldDataToParallelFile(w, fieldData=fieldData)

    w.closeGrid()
    w.save()
    return w.getFileName()


# ==============================================================================
def writeParallelVTKUnstructuredGrid(
    path,
    coordsDtype,
    sources,
    ghostlevel=0,
    cellData=None,
    pointData=None,
    fieldData=None,
    format="binary",
):
    """
    Writes a parallel VTK Unstructured Grid

    Parameters
    ----------
    path : str
        name of the file without extension.

    coordsdtype : dtype
        dtype of the coordinates.

    source : list
        list of the relative paths of the source files where the actual data is found

    ghostlevel : int, optional
        Number of ghost-levels by which
        the extents in the individual source files overlap.

    pointData : dict
        dictionnary containing the information about the arrays
        containing node centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)

    cellData :
        dictionnary containing the information about the arrays
        containing cell centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
    """
    # Get the extension + check that it's consistent accros all source files
    common_ext = sources[0].split(".")[-1]
    assert all(s.split(".")[-1] == common_ext for s in sources)
    if common_ext != "vtu":
        raise ValueError(
            f"Sources must be VTKUnstructuredGrid ('.vtu') and not {common_ext} files"
        )

    w = VtkParallelFile(path, VtkPUnstructuredGrid, format=format)
    w.openGrid(ghostlevel=ghostlevel)

    _addDataToParallelFile(w, cellData=cellData, pointData=pointData)

    w.openElement("PPoints")
    w.addPData("points", dtype=coordsDtype, ncomp=3)
    w.closeElement("PPoints")

    for source in sources:
        w.addPiece(source=source)

    _addFieldDataToParallelFile(w, fieldData=fieldData)

    w.closeGrid()
    w.save()
    return w.getFileName()
