""" High level API for parallel VTK files"""

from ..core.vtkfiles import (
    VtkParallelFile,
    VtkPImageData,
    VtkPPolyData,
    VtkPRectilinearGrid,
    VtkPStructuredGrid,
    VtkPUnstructuredGrid,
)

from ..utilities.utils import _addDataToParallelFile

__all__ = ['writeParallelVTKImageData',
           'writeParallelVTKGrid',
           'writeParallelVTKPolyData',
           'writeParallelVTKUnstructuredGrid']

# ==============================================================================
def writeParallelVTKImageData(
    path,
    starts, 
    ends, 
    sources,
    dimension,
    origin=(0., 0., 0.),
    spacing=(1., 1., 1.),  
    ghostlevel=0, 
    cellData=None, 
    pointData=None,
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
    
    dimension : 3-tuple or None, optional
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
        Number of cells which are present in neighbouring files.

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
    """
    # Check that every source as a start and an end
    assert len(starts) == len(ends) == len(sources)

    # Get the extension + check that it's consistent accros all source files
    common_ext = sources[0].split(".")[-1]
    assert all(s.split(".")[-1] == common_ext for s in sources), "All sources need to share the same extension"
    if common_ext != 'vti':
        raise ValueError(f"Sources must be VTKImageData ('.vti') and not {common_ext}")

    w = VtkParallelFile(path, VtkPImageData)
    start = (0, 0, 0)

    w.openGrid(start=start, end=dimension, origin=origin, spacing=spacing, ghostlevel=ghostlevel)

    _addDataToParallelFile(w, cellData=cellData, pointData=pointData)

    for start_source, end_source, source in zip(starts, ends, sources):
        w.addPiece(source, start_source, end_source)

    w.closeGrid()
    w.save()

    return w.getFileName()


# ==============================================================================
def writeParallelVTKGrid(
    path, 
    coordsData, 
    starts, 
    ends, 
    sources, 
    ghostlevel=0, 
    cellData=None, 
    pointData=None,
):
    """
    Writes a parallel vtk file from grid-like data:
    VTKStructuredGrid or VTKRectilinearGrid

    Parameters
    ----------
    path : str
        name of the file without extension.

    coordsData : tuple
        2-tuple (shape, dtype) where shape is the
        shape of the coordinates of the full mesh
        and dtype is the dtype of the coordinates.

    starts : list
        list of 3-tuple representing where each source file starts
        in each dimension.
    
    ends : list
        list of 3-tuple representing where each source file ends
        in each dimension

    source : list
        list of the relative paths of the source files where the actual data is found.

    ghostlevel : int, default=0
        Number of cells which are present in neighbouring files.

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
        raise ValueError("This functions is meant to work only with ")

    w = VtkParallelFile(path, ftype)
    start = (0, 0, 0)
    (s_x, s_y, s_z), dtype = coordsData
    end = s_x - 1, s_y - 1, s_z - 1

    w.openGrid(start=start, end=end, ghostlevel=ghostlevel)

    _addDataToParallelFile(w, cellData=cellData, pointData=pointData)

    if is_Rect:
        w.openElement("PCoordinates")
        w.addData("x_coordinates", dtype=dtype, ncomp=1)
        w.addData("y_coordinates", dtype=dtype, ncomp=1)
        w.addData("z_coordinates", dtype=dtype, ncomp=1)
        w.closeElement("PCoordinates")
    else:
        w.openElement("PPoints")
        w.addData("points", dtype=dtype, ncomp=3)
        w.closeElement("PPoints")

    for start_source, end_source, source in zip(starts, ends, sources):
        w.addPiece(source, start_source, end_source)

    w.closeGrid()
    w.save()
    return w.getFileName()


# ==============================================================================
def writeParallelVTKPolyData(
    path, 
    coordsdtype, 
    sources, 
    ghostlevel=0, 
    cellData=None, 
    pointData=None,
):
    """
    Writes a parallel VTK PolyData.

    Parameters
    ----------
    path : str
        name of the file without extension.

    coordsdtype : dtype
        dtype of the coordinates.

    source : list
        list of the relative paths of the source files where the actual data is found.

    ghostlevel : int, default=0
        Number of cells which are present in neighbouring files.

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
    
    """
    # Get the extension + check that it's consistent accros all source files
    common_ext = sources[0].split(".")[-1]
    assert all(s.split(".")[-1] == common_ext for s in sources)

    if common_ext != 'vtp':
        raise ValueError(f"Sources must be VTKPolyData ('.vtp') and not {common_ext}")
    
    w = VtkParallelFile(path, VtkPPolyData)

    w.openGrid(ghostlevel=ghostlevel)

    w.openElement('PPoints')
    w.addData('points', dtype=coordsdtype, ncomp=3)
    w.closeElement('PPoints')
    _addDataToParallelFile(w, cellData=cellData, pointData=pointData)

    for source in sources:
        w.addPiece(source)

    w.closeGrid()
    w.save()
    return w.getFileName()

# ==============================================================================
def writeParallelVTKUnstructuredGrid(
    path, 
    coordsdtype, 
    sources, 
    ghostlevel=0, 
    cellData=None, 
    pointData=None
):
    """
    Writes a parallel VTK Unstructured Grid 

    Parameters
    ----------
    path : str
        name of the file without extension.

    coordsdtype : dtype
        dtype of the coordinates.

    starts : list
        list of 3-tuple representing where each source file starts
        in each dimension

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
    """
    # Get the extension + check that it's consistent accros all source files
    common_ext = sources[0].split(".")[-1]
    assert all(s.split(".")[-1] == common_ext for s in sources)
    if common_ext != 'vtu':
        raise ValueError(f"Sources must be VTKPolyData ('.vtu') and not {common_ext}")
    
    w = VtkParallelFile(path, VtkPUnstructuredGrid)
    w.openGrid(ghostlevel=ghostlevel)

    _addDataToParallelFile(w, cellData=cellData, pointData=pointData)

    w.openElement("PPoints")
    w.addData("points", dtype=coordsdtype, ncomp=3)
    w.closeElement("PPoints")

    for source in sources:
        w.addPiece(source=source)

    w.closeGrid()
    w.save()
    return w.getFileName()
