# from mpi4py import MPI

# from ..api.parallel import (  # writeParallelVTKGrid,; writeParallelVTKImageData,; writeParallelVTKPolyData,
#    writeParallelVTKUnstructuredGrid,
# )

# from ..api.serial import gridToVTK, imageToVTK, polyLinesToVTK, unstructuredGridToVTK


# Helper functions
def get_data_info(cellData=None, pointData=None):
    """
    List the dtypes and number of components of
    the arrays in cell_data and point_data.
    Parameters
    ----------
    cellData : dict or None
        cell-centered data

    pointData : dict or None
        point-centered data

    Retuns
    ------
    cellData_info : dict
        Dtype and number of components of the cell centered data

    pointData_info : dict
        Dtype and number of components of the cell centered data
    """
    if cellData is None:
        cellData = {}
    if pointData is None:
        pointData = {}

    pointData_info = {}
    cellData_info = {}

    for name, data in cellData.items():
        if isinstance(data, tuple):
            cellData_info[name] = (data[0].dtype, 3)
        else:
            cellData_info[name] = (data.dtype, 1)

    for name, data in pointData.items():
        if isinstance(data, tuple):
            pointData_info[name] = (data[0].dtype, 3)
        else:
            pointData_info[name] = (data.dtype, 1)

    return cellData_info, pointData_info


# def parallelUnstructuredGridToVTK(
#     path,
#     x,
#     y,
#     z,
#     connectivity,
#     offsets,
#     cell_types,
#     cellData=None,
#     pointData=None,
#     fieldData=None,
#     comm=MPI.COMM_WORLD,
#     ghostlevel=0,
#     direct_format="ascii",
#     appended_format="raw",
#     compression=False,
#     compressor="zlib",
#     append=True,
#     check_cells=True,
# ):
#     """
#     Export one unstructured grid per rank
#     and one parallel unstructured grid on rank 0.

#     Parameters
#     ----------
#     path : str
#         name of the file without extension  or rank specific numbering where data should be saved.
#         Each rank will produce a file named ``filename + f".{rank}.vtu"``. Rank 0 will
#         also produce a parallel VTU file named ``filename.pvtu``

#     x : array-like
#         x coordinates of the vertices.

#     y : array-like
#         y coordinates of the vertices.

#     z : array-like
#         z coordinates of the vertices.

#     connectivity : array-like
#         1D array that defines the vertices associated to each element.
#         Together with offset define the connectivity or topology of the grid.
#         It is assumed that vertices in an element are listed consecutively.

#     offsets : array-like
#         1D array with the index of the last vertex of each element
#         in the connectivity array.
#         It should have length nelem,
#         where nelem is the number of cells or elements in the grid.

#     cell_types : array_like
#         1D array with an integer that defines the cell type of
#         each element in the grid.
#         It should have size nelem.
#         This should be assigned from evtk.vtk.VtkXXXX.tid, where XXXX represent
#         the type of cell.
#         Please check the VTK file format specification for allowed cell types.

#     cellData : dict, optional
#         dictionary with variables associated to each cell.
#         Keys should be the names of the variable stored in each array.
#         All arrays must have the same number of elements.

#     pointData : dict, optional
#         dictionary with variables associated to each vertex.
#         Keys should be the names of the variable stored in each array.
#         All arrays must have the same number of elements.

#     fieldData : dict, optional
#         dictionary with variables associated with the field.
#         Keys should be the names of the variable stored in each array.

#     comm : MPI.Intracomm, default=MPI.COMM_WORLD
#         Communicator.

#     ghotslevel : int default=0,
#         Number of cells which are shared between neighbouring files.

#     direct_format : str in {'ascii', 'binary'}, default='ascii'
#         how the data that isn't appended will be encoded.
#         If ``'ascii'``, the data will be human readable,
#         if ``'binary'`` it will use base 64
#         and can be compressed. See ``compressor`` argument.

#     appended_format : str in {'raw', 'binary'}, default='raw'
#         how that appended data will be encoded.
#         If ``'raw'``, raw binary data will be written to file.
#         This is space efficient and supported by vtk but isn't
#         valid XML. If ``'binary'``, data will be encoded using base64
#         and can be compressed. See ``compressor`` argument.

#     compression : Bool or int, default=False
#         compression level of the binary data.
#         Can be ``True``, ``False`` or any integer in ``[-1, 9]`` included.
#         If ``True``, compression will be set to -1 and use the default
#         value of the compressor.

#     compressor: str in {'zlib', 'lzma'}, default='zlib'
#         compression library to use for the binary data.

#     append : bool, default=True
#         Whether to write the data in appended mode or not.


#     check_cells : Bool, default=True
#         If True, checks ``cell_types`` and
#         ``offsets`` to ensure that the types are
#         correct and the number of points in each cell
#         is coherent with their type.
#         The understood cell types are listed in
#         ``py2vtk.core.vtkcells.py``.
#     """
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     unstructuredGridToVTK(
#         path + f".{rank}",
#         x,
#         y,
#         z,
#         connectivity=connectivity,
#         offsets=offsets,
#         cell_types=cell_types,
#         cellData=cellData,
#         pointData=pointData,
#         fieldData=fieldData,
#         direct_format=direct_format,
#         appended_format=appended_format,
#         compression=compression,
#         compressor=compressor,
#         append=append,
#         check_cells=check_cells,
#     )

#     if rank == 0:
#         cellData_info, pointData_info = get_data_info(cellData, pointData)
#         coords_dtype = x.dtype

#         writeParallelVTKUnstructuredGrid(
#             path,
#             coordsdtype=coords_dtype,
#             sources=[path + f".{rank}.vtu" for rank in range(size)],
#             cellData=cellData_info,
#             pointData=pointData_info,
#             ghostlevel=ghostlevel,
#         )
