import os

import numpy as np
from mpi4py import MPI

from ..api.parallel import (
    writeParallelVTKGrid,
    writeParallelVTKImageData,
    writeParallelVTKPolyData,
    writeParallelVTKUnstructuredGrid,
)
from ..api.serial import gridToVTK, imageToVTK, polyDataToVTK, unstructuredGridToVTK
from ..utilities.utils import get_data_info

__all__ = [
    "parallelImageToVTK",
    "parallelRectilinearGridToVTK",
    "parallelStructuredGridToVTK",
    "parallelPolyDataToVTK",
    "parallelUnstructuredGridToVTK",
]


# Helper functions
def _gather_starts_ends(local_start, local_end, comm):
    """
    Gather all of the local starts and ends on the root process of the communicator.

    This can be necessary for Image Data, Rectilinear Grid and Structured Grid.

    Parameters
    ----------
    local_start : tuple
        local start

    local_end : tuple
        local end

    comm : MPI.Intracomm
        Communicator
    """
    # Get the rank and size
    rank = comm.Get_rank()
    size = comm.Get_size()

    send_buf = np.zeros((2, 3), dtype="i")
    send_buf[0] = local_start
    send_buf[1] = local_end

    recv_buf = None

    if rank == 0:
        recv_buf = np.empty((size, 2, 3), dtype="i")

    comm.Gather(send_buf, recv_buf, root=0)

    return recv_buf


# API
def parallelImageToVTK(
    path,
    starts,
    ends=None,
    dimension=None,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    cellData=None,
    pointData=None,
    fieldData=None,
    comm=MPI.COMM_WORLD,
    ghostlevel=0,
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
):
    """
    Export one vtk image data per rank
    and one vtk parallel image data on rank 0.

    Parameters
    ----------

    path : str
        name of the file without extension  or rank specific numbering where data should be saved.
        Each rank will produce a file named ``filename + f".{rank}.vti"``. Rank 0 will
        also produce a parallel VTI file named ``filename.pvti``.

    starts : dict or tuple
        If ``starts`` is a dictionnary, it should map each rank to its start.
        If ``starts`` is a tuple, then it is assumed to be the start of the current rank.

    ends : dict or tuple, optional
        If ``ends`` is a dictionnary, it should map each rank to its end.
        If ``ends`` is a tuple, then it is assumed to be the end of the current rank.
        If ``ends`` is None it is deduced from the data.

    dimension : tuple or None, optional
        dimension of the image.

    origin : tuple, optional
        grid origin.
        The default is (0.0, 0.0, 0.0).

    spacing : tuple, optional
        grid spacing.
        The default is (1.0, 1.0, 1.0).

    cellData : dict, optional
        dictionary with variables associated to each cell.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.
        All arrays must have the same number of elements.

    pointData : dict, optional
        dictionary with variables associated to each vertex.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.
        All arrays must have the same number of elements.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.

    comm : MPI.Intracomm, default=MPI.COMM_WORLD
        Communicator.

    ghotslevel : int default=0,
        Number of cells which are shared between neighbouring files.

    direct_format : str in {'ascii', 'binary'}, default='ascii'
        how the data that isn't appended will be encoded.
        If ``'ascii'``, the data will be human readable,
        if ``'binary'`` it will use base 64
        and can be compressed. See ``compressor`` argument.

    appended_format : str in {'raw', 'binary'}, default='raw'
        how that appended data will be encoded.
        If ``'raw'``, raw binary data will be written to file.
        This is space efficient and supported by vtk but isn't
        valid XML. If ``'binary'``, data will be encoded using base64
        and can be compressed. See ``compressor`` argument.

    compression : Bool or int, default=False
        compression level of the binary data.
        Can be ``True``, ``False`` or any integer in ``[-1, 9]`` included.
        If ``True``, compression will be set to -1 and use the default
        value of the compressor.

    compressor: str in {'zlib', 'lzma'}, default='zlib'
        compression library to use for the binary data.

    append : bool, default=True
        Whether to write the data in appended mode or not.

    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if isinstance(starts, dict):
        start_rank = starts[rank]
    else:
        start_rank = starts

    if isinstance(ends, dict):
        end_rank = ends[rank]
    elif isinstance(ends, tuple):
        end_rank = ends
    elif ends is None:
        if cellData is not None:
            keys = list(cellData.keys())
            data = cellData[keys[0]]
            if hasattr(data, "shape"):
                end_rank = data.shape
            elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
                end_rank = data[0].shape
        elif pointData is not None:
            keys = list(pointData.keys())
            data = pointData[keys[0]]
            if hasattr(data, "shape"):
                end_rank = data.shape
            elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
                end_rank = data[0].shape
        end_rank = (
            start_rank[0] + end_rank[0] - 1,
            start_rank[1] + end_rank[1] - 1,
            start_rank[2] + end_rank[2] - 1,
        )

    imageToVTK(
        path + f".{rank}",
        start=start_rank,
        end=end_rank,
        origin=origin,
        spacing=spacing,
        cellData=cellData,
        pointData=pointData,
        fieldData=fieldData,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
        append=append,
    )

    if not (isinstance(starts, dict) and isinstance(ends, dict)):
        starts_ends = _gather_starts_ends(start_rank, end_rank, comm)
    else:
        starts_ends = np.empty((size, 2, 3), dtype="i")
        starts_ends[:, 0, :] = [starts[r] for r in range(size)]
        starts_ends[:, 1, :] = [ends[r] for r in range(size)]

    if rank == 0:
        cellData_info, pointData_info = get_data_info(cellData, pointData)
        path_base = os.path.basename(path)
        writeParallelVTKImageData(
            path,
            starts=starts_ends[:, 0, :],
            ends=starts_ends[:, 1, :],
            dimension=dimension,
            origin=origin,
            spacing=spacing,
            sources=[path_base + f".{rank}.vti" for rank in range(size)],
            cellData=cellData_info,
            pointData=pointData_info,
            fieldData=fieldData,
            ghostlevel=ghostlevel,
        )


def parallelRectilinearGridToVTK(
    path,
    x,
    y,
    z,
    starts,
    ends=None,
    dimension=None,
    cellData=None,
    pointData=None,
    fieldData=None,
    comm=MPI.COMM_WORLD,
    ghostlevel=0,
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
):
    """
    Export one vtk rectilinear grid per rank
    and one vtk parallel rectilinear grid on rank 0.

    Parameters
    ----------

    path : str
        name of the file without extension  or rank specific numbering where data should be saved.
        Each rank will produce a file named ``filename + f".{rank}.vtr"``. Rank 0 will
        also produce a parallel VTI file named ``filename.pvtr``.

    x : array-like
        x coordinates of the points..

    y : array-like
        y coordinates of the points..

    z : array-like
        z coordinates of the points..

    starts : dict or tuple
        If ``starts`` is a dictionnary, it should map each rank to its start.
        If ``starts`` is a tuple, then it is assumed to be the start of the current rank.

    ends : dict or tuple, optional
        If ``ends`` is a dictionnary, it should map each rank to its end.
        If ``ends`` is a tuple, then it is assumed to be the end of the current rank.
        If ``ends`` is None it is deduced from the data.

    dimension : tuple or None, optional
        dimension of the complete grid.

    cellData : dict, optional
        dictionary with variables associated to each cell.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.
        All arrays must have the same number of elements.

    pointData : dict, optional
        dictionary with variables associated to each vertex.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.
        All arrays must have the same number of elements.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.

    comm : MPI.Intracomm, default=MPI.COMM_WORLD
        Communicator.

    ghotslevel : int default=0,
        Number of cells which are shared between neighbouring files.

    direct_format : str in {'ascii', 'binary'}, default='ascii'
        how the data that isn't appended will be encoded.
        If ``'ascii'``, the data will be human readable,
        if ``'binary'`` it will use base 64
        and can be compressed. See ``compressor`` argument.

    appended_format : str in {'raw', 'binary'}, default='raw'
        how that appended data will be encoded.
        If ``'raw'``, raw binary data will be written to file.
        This is space efficient and supported by vtk but isn't
        valid XML. If ``'binary'``, data will be encoded using base64
        and can be compressed. See ``compressor`` argument.

    compression : Bool or int, default=False
        compression level of the binary data.
        Can be ``True``, ``False`` or any integer in ``[-1, 9]`` included.
        If ``True``, compression will be set to -1 and use the default
        value of the compressor.

    compressor: str in {'zlib', 'lzma'}, default='zlib'
        compression library to use for the binary data.

    append : bool, default=True
        Whether to write the data in appended mode or not.

    """
    if x.ndim == y.ndim == z.ndim == 3:
        raise ValueError(
            "x, y and z should be 1D arrays for the VTK Rectlinear Grid format.\n"
            "If x, y and z are meant to be 3D then 'parallelStructuredGridToVTK should be used instead."
        )

    rank = comm.Get_rank()
    size = comm.Get_size()

    if isinstance(starts, dict):
        start_rank = starts[rank]
    else:
        start_rank = starts

    if isinstance(ends, dict):
        end_rank = ends[rank]
    elif isinstance(ends, tuple):
        end_rank = ends
    elif ends is None:
        if cellData is not None:
            keys = list(cellData.keys())
            data = cellData[keys[0]]
            if hasattr(data, "shape"):
                end_rank = data.shape
            elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
                end_rank = data[0].shape
        elif pointData is not None:
            keys = list(pointData.keys())
            data = pointData[keys[0]]
            if hasattr(data, "shape"):
                end_rank = data.shape
            elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
                end_rank = data[0].shape
        end_rank = (
            start_rank[0] + end_rank[0] - 1,
            start_rank[1] + end_rank[1] - 1,
            start_rank[2] + end_rank[2] - 1,
        )

    gridToVTK(
        path + f".{rank}",
        x,
        y,
        z,
        start=start_rank,
        cellData=cellData,
        pointData=pointData,
        fieldData=fieldData,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
        append=append,
    )

    if not (isinstance(starts, dict) and isinstance(ends, dict)):
        starts_ends = _gather_starts_ends(start_rank, end_rank, comm)
    else:
        starts_ends = np.empty((size, 2, 3), dtype="i")
        starts_ends[:, 0, :] = [starts[r] for r in range(size)]
        starts_ends[:, 1, :] = [ends[r] for r in range(size)]

    if rank == 0:
        cellData_info, pointData_info = get_data_info(cellData, pointData)
        path_base = os.path.basename(path)
        writeParallelVTKGrid(
            path,
            starts=starts_ends[:, 0, :],
            ends=starts_ends[:, 1, :],
            coordsDtype=x.dtype,
            dimension=dimension,
            sources=[path_base + f".{rank}.vtr" for rank in range(size)],
            cellData=cellData_info,
            pointData=pointData_info,
            fieldData=fieldData,
            ghostlevel=ghostlevel,
            format=direct_format,
        )


def parallelStructuredGridToVTK(
    path,
    x,
    y,
    z,
    starts,
    ends=None,
    dimension=None,
    cellData=None,
    pointData=None,
    fieldData=None,
    comm=MPI.COMM_WORLD,
    ghostlevel=0,
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
):
    """
    Export one vtk structured grid per rank
    and one vtk parallel structured grid on rank 0.

    Parameters
    ----------

    path : str
        name of the file without extension  or rank specific numbering where data should be saved.
        Each rank will produce a file named ``filename + f".{rank}.vts"``. Rank 0 will
        also produce a parallel VTI file named ``filename.pvts``.

    x : array-like
        x coordinates of the points..

    y : array-like
        y coordinates of the points..

    z : array-like
        z coordinates of the points..

    starts : dict or tuple
        If ``starts`` is a dictionnary, it should map each rank to its start.
        If ``starts`` is a tuple, then it is assumed to be the start of the current rank.

    ends : dict or tuple, optional
        If ``ends`` is a dictionnary, it should map each rank to its end.
        If ``ends`` is a tuple, then it is assumed to be the end of the current rank.
        If ``ends`` is None it is deduced from the data.

    dimension : tuple or None, optional
        dimension of the complete grid.

    cellData : dict, optional
        dictionary with variables associated to each cell.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.
        All arrays must have the same number of elements.

    pointData : dict, optional
        dictionary with variables associated to each vertex.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.
        All arrays must have the same number of elements.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.

    comm : MPI.Intracomm, default=MPI.COMM_WORLD
        Communicator.

    ghotslevel : int default=0,
        Number of cells which are shared between neighbouring files.

    direct_format : str in {'ascii', 'binary'}, default='ascii'
        how the data that isn't appended will be encoded.
        If ``'ascii'``, the data will be human readable,
        if ``'binary'`` it will use base 64
        and can be compressed. See ``compressor`` argument.

    appended_format : str in {'raw', 'binary'}, default='raw'
        how that appended data will be encoded.
        If ``'raw'``, raw binary data will be written to file.
        This is space efficient and supported by vtk but isn't
        valid XML. If ``'binary'``, data will be encoded using base64
        and can be compressed. See ``compressor`` argument.

    compression : Bool or int, default=False
        compression level of the binary data.
        Can be ``True``, ``False`` or any integer in ``[-1, 9]`` included.
        If ``True``, compression will be set to -1 and use the default
        value of the compressor.

    compressor: str in {'zlib', 'lzma'}, default='zlib'
        compression library to use for the binary data.

    append : bool, default=True
        Whether to write the data in appended mode or not.

    """
    if x.ndim == y.ndim == z.ndim == 1:
        raise ValueError(
            "x, y and z should be 3D arrays for the VTK Structured Grid format.\n"
            "If x, y and z are meant to be 1D then 'parallelRectilinearGridToVTK should be used instead."
        )

    rank = comm.Get_rank()
    size = comm.Get_size()

    if isinstance(starts, dict):
        start_rank = starts[rank]
    else:
        start_rank = starts

    if isinstance(ends, dict):
        end_rank = ends[rank]
    elif isinstance(ends, tuple):
        end_rank = ends
    elif ends is None:
        if cellData is not None:
            keys = list(cellData.keys())
            data = cellData[keys[0]]
            if hasattr(data, "shape"):
                end_rank = data.shape
            elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
                end_rank = data[0].shape
        elif pointData is not None:
            keys = list(pointData.keys())
            data = pointData[keys[0]]
            if hasattr(data, "shape"):
                end_rank = data.shape
            elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
                end_rank = data[0].shape
        end_rank = (
            start_rank[0] + end_rank[0] - 1,
            start_rank[1] + end_rank[1] - 1,
            start_rank[2] + end_rank[2] - 1,
        )

    gridToVTK(
        path + f".{rank}",
        x,
        y,
        z,
        start=start_rank,
        cellData=cellData,
        pointData=pointData,
        fieldData=fieldData,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
        append=append,
    )

    if not (isinstance(starts, dict) and isinstance(ends, dict)):
        starts_ends = _gather_starts_ends(start_rank, end_rank, comm)
    else:
        starts_ends = np.empty((size, 2, 3), dtype="i")
        starts_ends[:, 0, :] = [starts[r] for r in range(size)]
        starts_ends[:, 1, :] = [ends[r] for r in range(size)]

    if rank == 0:
        cellData_info, pointData_info = get_data_info(cellData, pointData)
        path_base = os.path.basename(path)
        writeParallelVTKGrid(
            path,
            starts=starts_ends[:, 0, :],
            ends=starts_ends[:, 1, :],
            coordsDtype=x.dtype,
            dimension=dimension,
            sources=[path_base + f".{rank}.vts" for rank in range(size)],
            cellData=cellData_info,
            pointData=pointData_info,
            fieldData=fieldData,
            ghostlevel=ghostlevel,
            format=direct_format,
        )


def parallelPolyDataToVTK(
    path,
    x,
    y,
    z,
    vertices=None,
    lines=None,
    strips=None,
    polys=None,
    cellData=None,
    pointData=None,
    fieldData=None,
    comm=MPI.COMM_WORLD,
    ghostlevel=0,
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
):
    """
    Export one vtk polydata per rank
    and one vtk parallel polydata on rank 0.

    Parameters
    ----------

    path : str
        name of the file without extension where data should be saved.

    x : array-like
        x coordinates of the points.

    y : array-like
        y coordinates of the points.

    z : array-like
        z coordinates of the points.

    vertices : array-like or None, optional
        1-D array containing the index of the points which should be saved as vertices.

    lines : 2-tuple of array-likes or list of array-likes or None, optional
        If a 2-tuple or array-likes, should be (connectivity, offsets)
        where connectivity should defines the points associated to each line and
        offsets should define the index of the last point in each cell (here line).
        If a list of array-likes, each element in the list should define the points associated
        to each line.

    strips : 2-tuple of array-likes or list of array-likes or None, optional
        If a 2-tuple or array-likes, should be (connectivity, offsets)
        where connectivity should defines the points associated to each strip and
        offsets should define the index of the last point in each cell (here strip).
        If a list of array-likes, each element in the list should define the points associated
        to each strip.

    polys : 2-tuple of array-likes or list of array-likes or None, optional
        If a 2-tuple or array-likes, should be (connectivity, offsets)
        where connectivity should defines the points associated to each polygon and
        offsets should define the index of the last point in each cell (here polygon).
        If a list of array-likes, each element in the list should define the points associated
        to each polygon.

    cellData : dict or 4-tuple of dicts, optional
        dictionary containing cell centered data or tuple of 4 dictionaries,
        one for each cell type (vertices, lines, strips and polys).
        Keys should be the names of the data arrays.
        Values should be arrays or 3-tuple of arrays.
        Arrays must have the same dimensions in all directions and
        must only contain scalar data.

    pointData : dict, optional
        dictionary with variables associated to each vertex.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.
        All arrays must have the same number of elements.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.

    comm : MPI.Intracomm, default=MPI.COMM_WORLD
        Communicator.

    ghotslevel : int default=0,
        Number of cells which are shared between neighbouring files.

    direct_format : str in {'ascii', 'binary'}, default='ascii'
        how the data that isn't appended will be encoded.
        If ``'ascii'``, the data will be human readable,
        if ``'binary'`` it will use base 64
        and can be compressed. See ``compressor`` argument.

    appended_format : str in {'raw', 'binary'}, default='raw'
        how that appended data will be encoded.
        If ``'raw'``, raw binary data will be written to file.
        This is space efficient and supported by vtk but isn't
        valid XML. If ``'binary'``, data will be encoded using base64
        and can be compressed. See ``compressor`` argument.

    compression : Bool or int, default=False
        compression level of the binary data.
        Can be ``True``, ``False`` or any integer in ``[-1, 9]`` included.
        If ``True``, compression will be set to -1 and use the default
        value of the compressor.

    compressor: str in {'zlib', 'lzma'}, default='zlib'
        compression library to use for the binary data.

    append : bool, default=True
        Whether to write the data in appended mode or not.

    Notes
    -----

    While Vtk PolyData does support cell-centered data, the way it does is not
    intuitive as the cell are numbered globally across each cell type and ordered in the following way:
    verts, lines, polys and strips. On top of that, for what is still a mistery to me, cell data written in base64
    is read improperly (despite being written properly) and shows wrong results in paraview and when read using the
    python vtk library. For this reason, when provided with cell-centered data, this function will enforce 'raw'
    as the appended format and 'ascii' as the direct format.

    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    polyDataToVTK(
        path + f".{rank}",
        x,
        y,
        z,
        vertices=vertices,
        lines=lines,
        strips=strips,
        polys=polys,
        cellData=cellData,
        pointData=pointData,
        fieldData=fieldData,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
        append=append,
    )

    if rank == 0:
        cellData_info, pointData_info = get_data_info(cellData, pointData)
        coords_dtype = x.dtype
        path_base = os.path.basename(path)
        writeParallelVTKPolyData(
            path,
            coordsDtype=coords_dtype,
            sources=[path_base + f".{rank}.vtp" for rank in range(size)],
            cellData=cellData_info,
            pointData=pointData_info,
            fieldData=fieldData,
            ghostlevel=ghostlevel,
            format=direct_format,
        )


def parallelUnstructuredGridToVTK(
    path,
    x,
    y,
    z,
    connectivity,
    offsets,
    cell_types,
    faces=None,
    faceoffsets=None,
    cellData=None,
    pointData=None,
    fieldData=None,
    comm=MPI.COMM_WORLD,
    ghostlevel=0,
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
    check_cells=True,
):
    """
    Export one unstructured grid per rank
    and one vtk parallel unstructured grid on rank 0.

    Parameters
    ----------

    path : str
        name of the file without extension  or rank specific numbering where data should be saved.
        Each rank will produce a file named ``filename + f".{rank}.vtu"``. Rank 0 will
        also produce a parallel VTU file named ``filename.pvtu``.

    x : array-like
        x coordinates of the vertices.

    y : array-like
        y coordinates of the vertices.

    z : array-like
        z coordinates of the vertices.

    connectivity : array-like
        1D array that defines the vertices associated to each element.
        Together with offset define the connectivity or topology of the grid.
        It is assumed that vertices in an element are listed consecutively.

    offsets : array-like
        1D array with the index of the last vertex of each element
        in the connectivity array.
        It should have length nelem,
        where nelem is the number of cells or elements in the grid.

    cell_types : array_like
        1D array with an integer that defines the cell type of
        each element in the grid.
        It should have size nelem.
        This should be filed using py2vtk.core.vtkcells.VtkXXXX.tid, where XXXX represent
        the type of cell.
        Please check the VTK file format specification or py2vtk.core.vtkcells.py for allowed cell types.

    faces : array_like or None, optional
        1D integer array describing the faces of polyhedric cells.
        This is only required and used if there are polyhedra in the grid (cell id 42).
        When used it is expected to be formatted in the following way for each polyhedron:
        Number of faces, Number of points in face 0, first point of face 0, ... and so on.

    faceoffsets : array_like or None, optional
        1D integer array with the index of the last vertex of each polyhedron in the faces array.

    cellData : dict, optional
        dictionary with variables associated to each cell.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.
        All arrays must have the same number of elements.

    pointData : dict, optional
        dictionary with variables associated to each vertex.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.
        All arrays must have the same number of elements.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.

    comm : MPI.Intracomm, default=MPI.COMM_WORLD
        Communicator.

    ghotslevel : int default=0,
        Number of cells which are shared between neighbouring files.

    direct_format : str in {'ascii', 'binary'}, default='ascii'
        how the data that isn't appended will be encoded.
        If ``'ascii'``, the data will be human readable,
        if ``'binary'`` it will use base 64
        and can be compressed. See ``compressor`` argument.

    appended_format : str in {'raw', 'binary'}, default='raw'
        how that appended data will be encoded.
        If ``'raw'``, raw binary data will be written to file.
        This is space efficient and supported by vtk but isn't
        valid XML. If ``'binary'``, data will be encoded using base64
        and can be compressed. See ``compressor`` argument.

    compression : Bool or int, default=False
        compression level of the binary data.
        Can be ``True``, ``False`` or any integer in ``[-1, 9]`` included.
        If ``True``, compression will be set to -1 and use the default
        value of the compressor.

    compressor: str in {'zlib', 'lzma'}, default='zlib'
        compression library to use for the binary data.

    append : bool, default=True
        Whether to write the data in appended mode or not.

    check_cells : Bool, default=True
        If True, checks ``cell_types`` and
        ``offsets`` to ensure that the types are
        correct and the number of points in each cell
        is coherent with their type.
        The understood cell types are listed in
        ``py2vtk.core.vtkcells.py``.

    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    unstructuredGridToVTK(
        path + f".{rank}",
        x,
        y,
        z,
        connectivity=connectivity,
        offsets=offsets,
        cell_types=cell_types,
        faces=faces,
        faceoffsets=faceoffsets,
        cellData=cellData,
        pointData=pointData,
        fieldData=fieldData,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
        append=append,
        check_cells=check_cells,
    )

    if rank == 0:
        cellData_info, pointData_info = get_data_info(cellData, pointData)
        coords_dtype = x.dtype
        path_base = os.path.basename(path)
        writeParallelVTKUnstructuredGrid(
            path,
            coordsDtype=coords_dtype,
            sources=[path_base + f".{rank}.vtu" for rank in range(size)],
            cellData=cellData_info,
            pointData=pointData_info,
            fieldData=fieldData,
            ghostlevel=ghostlevel,
            format=direct_format,
        )
