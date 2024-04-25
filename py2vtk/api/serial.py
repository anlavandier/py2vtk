# ******************************************************************************
# * Copyright 2010 - 2016 Paulo A. Herrera. All rights reserved.               *
# *                                                                            *
# * Redistribution and use in source and binary forms, with or without         *
# * modification, are permitted provided that the following conditions         *
# * are met:                                                                   *
# *                                                                            *
# *  1. Redistributions of source code must retain the above copyright notice, *
# *  this list of conditions and the following disclaimer.                     *
# *                                                                            *
# *  2. Redistributions in binary form must reproduce the above copyright      *
# *  notice, this list of conditions and the following disclaimer in the       *
# *  documentation and/or other materials provided with the distribution.      *
# *                                                                            *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES  *
# * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN *
# * NO EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,*
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES         *
# * (INCLUDING, BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES;*
# * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND*
# * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT *
# * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF   *
# * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          *
# ******************************************************************************
""" High level API for serial VTK Files"""

import warnings

import numpy as np

from ..core.vtkcells import (
    Vtk_points_per_cell,
    VtkLine,
    VtkPixel,
    VtkPolyLine,
    VtkVertex,
)
from ..core.xml.utils import _addDataToFile, _addFieldDataToFile
from ..core.xml.vtkfiles import (
    VtkFile,
    VtkImageData,
    VtkPolyData,
    VtkRectilinearGrid,
    VtkStructuredGrid,
    VtkUnstructuredGrid,
)

__all__ = [
    "imageToVTK",
    "gridToVTK",
    "polyDataToVTK",
    "pointsToVTK",
    "linesToVTK",
    "polyLinesToVTK",
    "unstructuredGridToVTK",
    "cylinderToVTK",
]


# ==============================================================================
def imageToVTK(
    path,
    start=(0, 0, 0),
    end=None,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    cellData=None,
    pointData=None,
    fieldData=None,
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
):
    """
    Export data values as a rectangular image.

    Parameters
    ----------

    path : str
        name of the file without extension where data should be saved.

    start : tuple, optional
        start of this image relative to a global image.
        Used in the distributed context where each process
        writes its own vtk file. Default is (0, 0, 0).

    end : tuple or None, optional
        end of this image relative to a global image.
        Used in the distributed context where each process
        writes its own vtk file. If None, it will be deduced
        from the data.

    origin : tuple, optional
        grid origin.
        The default is (0.0, 0.0, 0.0).

    spacing : tuple, optional
        grid spacing.
        The default is (1.0, 1.0, 1.0).

    cellData : dict, optional
        dictionary containing arrays with cell centered data.
        Keys should be the names of the data arrays.
        Arrays must have the same dimensions in all directions and can contain
        scalar data ([n,n,n]) or vector data ([n,n,n],[n,n,n],[n,n,n]).
        The default is None.

    pointData : dict, optional
        dictionary containing arrays with node centered data.
        Keys should be the names of the data arrays.
        Arrays must have same dimension in each direction and
        they should be equal to the dimensions of the cell data plus one and
        can contain scalar data ([n+1,n+1,n+1]) or
        ([n+1,n+1,n+1],[n+1,n+1,n+1],[n+1,n+1,n+1]).
        The default is None.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

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

    Returns
    -------

    str
        Full path to saved file.

    Notes
    -----

    At least, cellData or pointData must be present
    to infer the dimensions of the image.

    """
    assert cellData is not None or pointData is not None

    # Extract dimensions
    if end is None:
        if cellData is not None:
            keys = list(cellData.keys())
            data = cellData[keys[0]]
            if hasattr(data, "shape"):
                end = data.shape
            elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
                end = data[0].shape
        elif pointData is not None:
            keys = list(pointData.keys())
            data = pointData[keys[0]]
            if hasattr(data, "shape"):
                end = data.shape
            elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
                end = tuple(np.array(data[0].shape) - 1)
        end = (start[0] + end[0], start[1] + end[1], start[2] + end[2])

    # Write data to file
    w = VtkFile(
        path,
        VtkImageData,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
    )

    w.openGrid(start=start, end=end, origin=origin, spacing=spacing)
    w.openPiece(start=start, end=end)
    _addDataToFile(w, cellData, pointData, append=append)
    w.closePiece()
    _addFieldDataToFile(w, fieldData, append=append)
    w.closeGrid()
    w.save()
    return w.getFileName()


# ==============================================================================
def gridToVTK(
    path,
    x,
    y,
    z,
    start=(0, 0, 0),
    cellData=None,
    pointData=None,
    fieldData=None,
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
):
    """
    Write data values as a rectilinear or structured grid.

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

    start : tuple, optional
        start of this grid relative to a global grid.
        Used in the distributed context where each process
        writes its own vtk file. Default is (0, 0, 0).

    cellData : dict, optional
        dictionary containing arrays with cell centered data.
        Keys should be the names of the data arrays.
        Arrays must have the same dimensions in all directions and must contain
        only scalar data.

    pointData : dict, optional
        dictionary containing arrays with node centered data.
        Keys should be the names of the data arrays.
        Arrays must have same dimension in each direction and
        they should be equal to the dimensions of the cell data plus one and
        must contain only scalar data.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

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

    Returns
    -------

    str
        Full path to saved file.

    Notes
    -----

    Coordinates can be 1D or 3D depending if the grid should
    be saved as a rectilinear or logically structured grid,
    respectively.
    Arrays should contain coordinates of the nodes of the grid.
    If arrays are 1D, then the grid should be Cartesian,
    i.e. faces in all cells are orthogonal.
    If arrays are 3D, then the grid should be logically structured
    with hexahedral cells.
    In both cases the arrays dimensions should be
    equal to the number of nodes of the grid.

    """
    nx = ny = nz = 0

    if x.ndim == 1 and y.ndim == 1 and z.ndim == 1:
        nx, ny, nz = x.size - 1, y.size - 1, z.size - 1
        isRect = True
        ftype = VtkRectilinearGrid
    elif x.ndim == 3 and y.ndim == 3 and z.ndim == 3:
        s = x.shape
        nx, ny, nz = s[0] - 1, s[1] - 1, s[2] - 1
        isRect = False
        ftype = VtkStructuredGrid
    else:
        raise ValueError(
            f"x, y and z should have ndim == 3 or 1"
            f" but they have ndim of {x.ndim}, {y.ndim}"
            f" and {z.ndim} respectively"
        )

    # Write extent
    end = (start[0] + nx, start[1] + ny, start[2] + nz)

    # Open File
    w = VtkFile(
        path,
        ftype,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
    )

    # Open Grid part
    w.openGrid(start=start, end=end)
    w.openPiece(start=start, end=end)

    # Add coordinates
    if isRect:
        w.openElement("Coordinates")
        w.addData("x_coordinates", x, append=append)
        w.addData("y_coordinates", y, append=append)
        w.addData("z_coordinates", z, append=append)
        w.closeElement("Coordinates")
    else:
        w.openElement("Points")
        w.addData("points", (x, y, z), append=append)
        w.closeElement("Points")

    # Add data
    _addDataToFile(w, cellData, pointData, append=append)

    # Close Grid part
    w.closePiece()

    _addFieldDataToFile(w, fieldData, append=append)

    w.closeGrid()

    # Close file
    w.save()

    return w.getFileName()


# ==============================================================================
def polyDataToVTK(
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
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
):
    """
    Write vertices, lines, strips and polygons as a VTK Polydata

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
        1-D array containing the index of the points
        which should be saved as vertices.

    lines : 2-tuple of array-likes or list of array-likes or None, optional
        If a 2-tuple or array-likes, should be (connectivity, offsets)
        where connectivity should defines the points associated to each line
        and offsets should define the index of the last point in each cell
        (here line). If a list of array-likes, each element in the list should
        define the points associated to each line.

    strips : 2-tuple of array-likes or list of array-likes or None, optional
        If a 2-tuple or array-likes, should be (connectivity, offsets)
        where connectivity should defines the points associated to each
        strip and offsets should define the index of the last point in each
        cell (here strip). If a list of array-likes, each element in the list
        should define the points associated to each strip.

    polys : 2-tuple of array-likes or list of array-likes or None, optional
        If a 2-tuple or array-likes, should be (connectivity, offsets)
        where connectivity should defines the points associated to each
        polygon and offsets should define the index of the last point in each
        cell (here polygon). If a list of array-likes, each element in the
        list should define the points associated to each polygon.

    cellData : dict or 4-tuple of dicts, optional
        dictionary containing cell centered data or tuple of 4 dictionaries,
        one for each cell type (vertices, lines, strips and polys).
        Keys should be the names of the data arrays.
        Values should be arrays or 3-tuple of arrays.
        Arrays must have the same dimensions in all directions and
        must only contain scalar data.

    pointData : dict, optional
        dictionary containing node centered data.
        Keys should be the names of the data arrays.
        Values should be arrays or 3-tuple of arrays.
        Arrays must have same dimension in each direction and
        must contain only scalar data.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

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

    Returns
    -------

    str
        Full path to saved file.

    Notes
    -----

    While Vtk PolyData does support cell-centered data, the way it does is not
    intuitive as the cell are numbered globally across each cell type and
    ordered in the following way: verts, lines, polys and strips.
    On top of that, for what is still a mistery to me,
    cell data written in base64 is read improperly (despite being written
    properly) and shows wrong results in paraview and when read using the
    python vtk library. For this reason, when provided with cell-centered data,
    this function will enforce 'raw' as the appended format and 'ascii'
    as the direct format.

    Warns
    -----

    UserWarning
        If cellData is not None and the appended or direct format is binary.
    """
    assert x.size == y.size == z.size
    npoints = x.size

    nverts = 0
    if vertices is not None:
        nverts = len(vertices)
        vertices_conn = np.array(vertices, dtype="int32")
        vertices_off = np.arange(1, nverts + 1)

    nlines = 0
    if lines is not None:
        if isinstance(lines, (tuple)):
            assert len(lines) == 2
            lines_conn, lines_off = lines
            nlines = len(lines_off)
        elif isinstance(lines, list):
            nlines = len(lines)
            lines_off = np.zeros(nlines, dtype="int32")
            current_offset = 0
            for i, line in enumerate(lines):
                current_offset += len(line)
                lines_off[i] = current_offset

            lines_conn = np.concatenate(lines, dtype="int32")

    nstrips = 0
    if strips is not None:
        if isinstance(strips, (tuple)):
            assert len(strips) == 2
            strips_conn, strips_off = strips
            nstrips = len(strips_off)
        elif isinstance(strips, list):
            nstrips = len(strips)
            strips_off = np.zeros(nstrips, dtype="int32")
            current_offset = 0
            for i, strip in enumerate(strips):
                current_offset += len(strip)
                strips_off[i] = current_offset

            strips_conn = np.concatenate(strips, dtype="int32")

    npolys = 0
    if polys is not None:
        if isinstance(polys, (tuple)):
            assert len(polys) == 2
            polys_conn, polys_off = polys
            npolys = len(polys_off)
        elif isinstance(polys, list):
            npolys = len(polys)
            polys_off = np.zeros(npolys, dtype="int32")
            current_offset = 0
            for i, poly in enumerate(polys):
                current_offset += len(poly)
                polys_off[i] = current_offset

            polys_conn = np.concatenate(polys, dtype="int32")

    if cellData is not None and (
        appended_format == "binary" or direct_format == "binary"
    ):
        warnings.warn(
            "Cell Data written in base64 will be improperly read by Paraview "
            " and the python vtk library.\n"
            " Formats are set to 'raw' and 'ascii'"
        )
        direct_format = "ascii"
        appended_format = "raw"

    w = VtkFile(
        path,
        VtkPolyData,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
    )

    w.openGrid()
    w.openPiece(
        npoints=npoints, nverts=nverts, nlines=nlines, nstrips=nstrips, npolys=npolys
    )

    w.openElement("Points")
    w.addData("points", (x, y, z), append=append)
    w.closeElement("Points")
    w.openElement("Verts")
    if nverts != 0:
        w.addData("connectivity", data=vertices_conn, append=append)
        w.addData("offsets", data=vertices_off, append=append)
    w.closeElement("Verts")

    w.openElement("Lines")
    if nlines != 0:
        w.addData("connectivity", data=lines_conn, append=append)
        w.addData("offsets", data=lines_off, append=append)
    w.closeElement("Lines")

    w.openElement("Strips")
    if nstrips != 0:
        w.addData("connectivity", data=strips_conn, append=append)
        w.addData("offsets", data=strips_off, append=append)
    w.closeElement("Strips")

    w.openElement("Polys")
    if npolys != 0:
        w.addData("connectivity", data=polys_conn, append=append)
        w.addData("offsets", data=polys_off, append=append)
    w.closeElement("Polys")

    if isinstance(cellData, tuple):
        # Cell data specified per cell type, extend with np.nan
        # Cell are treated in the order of verts, lines, polys, strips
        # according to this blog post https://narkive.com/hQoDBjCE.3 and
        # testing on my end
        assert len(cellData) == 4

        (celldata_verts, celldata_lines, celldata_strips, celldata_polys) = cellData

        cellData = {}

        for name, data in celldata_verts:
            if isinstance(data, tuple):
                cellData[name] = tuple(
                    np.concatenate([data_i, np.full(nlines + nstrips + npolys, np.nan)])
                    for data_i in data
                )
            else:
                cellData[name] = np.concatenate(
                    [data, np.full(nlines + nstrips + npolys, np.nan)]
                )

        for name, data in celldata_lines:
            if isinstance(data, tuple):
                cellData[name] = tuple(
                    np.concatenate(
                        [
                            np.full(nverts, np.nan),
                            data_i,
                            np.full(nstrips + npolys, np.nan),
                        ]
                    )
                    for data_i in data
                )
            else:
                cellData[name] = np.concatenate(
                    [np.full(nverts, np.nan), data, np.full(nstrips + npolys, np.nan)]
                )

        for name, data in celldata_polys:
            if isinstance(data, tuple):
                cellData[name] = tuple(
                    np.concatenate(
                        [
                            np.full(nverts + nlines, np.nan),
                            data_i,
                            np.full(nstrips, np.nan),
                        ]
                    )
                    for data_i in data
                )
            else:
                cellData[name] = np.concatenate(
                    [np.full(nverts + nlines, np.nan), data, np.full(nstrips, np.nan)]
                )

        for name, data in celldata_strips:
            if isinstance(data, tuple):
                cellData[name] = tuple(
                    np.concatenate([np.full(nverts + nlines + npolys, np.nan), data_i])
                    for data_i in data
                )
            else:
                cellData[name] = np.concatenate(
                    [np.full(nverts + nlines + npolys, np.nan), data]
                )

    _addDataToFile(w, cellData=cellData, pointData=pointData, append=append)

    w.closePiece()
    _addFieldDataToFile(w, fieldData, append=append)
    w.closeGrid()

    w.save()
    return w.getFileName()


# ==============================================================================
def pointsToVTK(
    path,
    x,
    y,
    z,
    data=None,
    fieldData=None,
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
):
    """
    Export points and associated data as an unstructured grid.

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

    data : dict, optional
        dictionary with variables associated to each point.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

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

    Returns
    -------

    str
        Full path to saved file.

    """

    assert x.size == y.size == z.size
    npoints = x.size

    # create some temporary arrays to write grid topology
    offsets = np.arange(
        start=1, stop=npoints + 1, dtype="int32"
    )  # index of last node in each cell
    connectivity = np.arange(
        npoints, dtype="int32"
    )  # each point is only connected to itself
    cell_types = np.empty(npoints, dtype="uint8")

    cell_types[:] = VtkVertex.tid

    w = VtkFile(
        path,
        VtkUnstructuredGrid,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
    )

    w.openGrid()
    w.openPiece(ncells=npoints, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z), append=append)
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity, append=append)
    w.addData("offsets", offsets, append=append)
    w.addData("types", cell_types, append=append)
    w.closeElement("Cells")

    _addDataToFile(w, cellData=None, pointData=data, append=append)

    w.closePiece()
    _addFieldDataToFile(w, fieldData, append=append)
    w.closeGrid()

    w.save()
    return w.getFileName()


# ==============================================================================
def linesToVTK(
    path,
    x,
    y,
    z,
    cellData=None,
    pointData=None,
    fieldData=None,
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
):
    """
    Export line segments that joint 2 points and associated data.

    Parameters
    ----------
    path : str
        name of the file without extension where data should be saved.

    x : array-like
        x coordinates of the points in lines.

    y : array-like
        y coordinates of the points in lines.

    z : array-like
        z coordinates of the points in lines.

    cellData : dict, optional
        dictionary with variables associated to each line.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.

    pointData : dict, optional
        dictionary containing node centered data.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

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

    Returns
    -------

    str
        Full path to saved file.

    Notes
    -----

    x, y, z are 1D arrays with coordinates of the vertex of the lines.
    It is assumed that each line is defined by two points,
    then the length of the arrays should be equal to 2 * number of lines.

    """
    assert x.size == y.size == z.size
    assert x.size % 2 == 0
    npoints = x.size
    ncells = x.size / 2

    # create some temporary arrays to write grid topology
    offsets = np.arange(
        start=2, step=2, stop=npoints + 1, dtype="int32"
    )  # index of last node in each cell
    connectivity = np.arange(
        npoints, dtype="int32"
    )  # each point is only connected to itself
    cell_types = np.empty(npoints, dtype="uint8")

    cell_types[:] = VtkLine.tid

    w = VtkFile(
        path,
        VtkUnstructuredGrid,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
    )
    w.openGrid()
    w.openPiece(ncells=ncells, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z), append=append)
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity, append=append)
    w.addData("offsets", offsets, append=append)
    w.addData("types", cell_types, append=append)
    w.closeElement("Cells")

    _addDataToFile(w, cellData=cellData, pointData=pointData, append=append)

    w.closePiece()
    _addFieldDataToFile(w, fieldData, append=append)
    w.closeGrid()

    w.save()
    return w.getFileName()


# ==============================================================================
def polyLinesToVTK(
    path,
    x,
    y,
    z,
    pointsPerLine,
    cellData=None,
    pointData=None,
    fieldData=None,
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
):
    """
    Export line segments that joint n points and associated data.

    Parameters
    ----------

    path : str
        name of the file without extension where data should be saved.

    x : array-like
        x coordinates of the points in lines.

    y : array-like
        y coordinates of the points in lines.

    z : array-like
        z coordinates of the points in lines.

    pointsPerLine : array-like
        Points in each poly-line.

    cellData : dict, optional
        dictionary with variables associated to each line.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.

    pointData : dict, optional
        dictionary containing node centered data.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

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

    Returns
    -------

    str
        Full path to saved file.

    """
    assert x.size == y.size == z.size
    npoints = x.size
    ncells = pointsPerLine.size

    # create some temporary arrays to write grid topology
    # index of last node in each cell
    offsets = np.zeros(ncells, dtype="int32")
    ii = 0
    for i in range(ncells):
        ii += pointsPerLine[i]
        offsets[i] = ii

    connectivity = np.arange(
        npoints, dtype="int32"
    )  # each line connects points that are consecutive

    cell_types = np.empty(npoints, dtype="uint8")
    cell_types[:] = VtkPolyLine.tid

    w = VtkFile(
        path,
        VtkUnstructuredGrid,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
    )

    w.openGrid()
    w.openPiece(ncells=ncells, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z), append=append)
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity, append=append)
    w.addData("offsets", offsets, append=append)
    w.addData("types", cell_types, append=append)
    w.closeElement("Cells")

    _addDataToFile(w, cellData=cellData, pointData=pointData, append=append)

    w.closePiece()
    _addFieldDataToFile(w, fieldData, append=append)
    w.closeGrid()

    w.save()
    return w.getFileName()


# ==============================================================================
def unstructuredGridToVTK(
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
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
    check_cells=True,
):
    """
    Export unstructured grid and associated data.

    Parameters
    ----------

    path : str
        name of the file without extension where data should be saved.

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
        This should be assigned from evtk.vtk.VtkXXXX.tid, where XXXX represent
        the type of cell.
        Please check the VTK file format specification or py2vtk.core.vtkcells
        for allowed cell types.

    faces : array_like or None, optional
        1D integer array describing the faces of polyhedric cells.
        This is only required and used if there are polyhedra in the
        grid (cell id 42). When used it is expected to be formatted in the
        following way for each polyhedron:
        Number of faces, Number of points in face 0, first point of face 0,
        ... and so on.

    faceoffsets : array_like or None, optional
        1D integer array with the index of the last vertex of each polyhedron
        in the faces array.

    cellData : dict, optional
        dictionary containing cell centered data.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.
        All arrays must have the same number of elements.

    pointData : dict, optional
        dictionary containing node centered data.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.
        All arrays must have the same number of elements.

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
        Values should be arrays or 3-tuple of arrays.

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

    Returns
    -------

    str
        Full path to saved file.

    """
    assert x.size == y.size == z.size
    npoints = x.size
    ncells = cell_types.size
    assert offsets.size == ncells

    if check_cells:
        n_points_type_0 = Vtk_points_per_cell.get(cell_types[0])
        if n_points_type_0 is None:
            raise ValueError(
                f"Cell type {cell_types[0]} is not recognized."
                "If this is a correct VTK cell type, please raise an issue"
                "on github."
            )
        if n_points_type_0 != -1 and n_points_type_0 != offsets[0]:
            raise ValueError(
                "Incorrect number of points in cell 0."
                f"{n_points_type_0} points were expected"
                f" but {offsets[0]} were given"
            )
        for i in range(1, ncells):
            n_points_type_i = Vtk_points_per_cell.get(cell_types[i])
            if n_points_type_i is None:
                raise ValueError(
                    f"Cell type {cell_types[0]} is not recognized."
                    "If this is a correct VTK cell type, please raise an issue"
                    "on github."
                )
            if n_points_type_i != -1 and n_points_type_i != (
                offsets[i] - offsets[i - 1]
            ):
                raise ValueError(
                    "Incorrect number of points in cell 0. "
                    f" {n_points_type_0} points were expected"
                    f" but {offsets[i]} were given"
                )

        if 42 in cell_types:
            if faces is None or faceoffsets is None:
                raise ValueError(
                    "Polyhedric cells (cell id 42) require"
                    " both faces and faces_offsets"
                )

    w = VtkFile(
        path,
        VtkUnstructuredGrid,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
    )

    w.openGrid()
    w.openPiece(ncells=ncells, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z), append=append)
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity, append=append)
    w.addData("offsets", offsets, append=append)
    w.addData("types", cell_types, append=append)
    if faces is not None and faceoffsets is not None:
        w.addData("faces", faces, append=append)
        w.addData("faceoffsets", faceoffsets, append=append)
    w.closeElement("Cells")

    _addDataToFile(w, cellData=cellData, pointData=pointData, append=append)

    w.closePiece()
    _addFieldDataToFile(w, fieldData, append=append)
    w.closeGrid()

    w.save()
    return w.getFileName()


# ==============================================================================
def cylinderToVTK(
    path,
    x0,
    y0,
    z0,
    z1,
    radius,
    nlayers,
    npilars=16,
    cellData=None,
    pointData=None,
    fieldData=None,
    direct_format="ascii",
    appended_format="raw",
    compression=False,
    compressor="zlib",
    append=True,
):
    """
    Export cylinder as VTK unstructured grid.

    Parameters
    ----------

    path : str
        name of the file without extension where data should be saved.

    x0 : float
        x-center of the cylinder.

    y0 : float
        y-center of the cylinder.

    z0 : float
        lower end of the cylinder.

    z1 : float
        upper end of the cylinder.

    radius : float
        radius of the cylinder.

    nlayers : int
        Number of layers in z direction to divide the cylinder.

    npilars : int, optional
        Number of points around the diameter of the cylinder.
        Higher value gives higher resolution to represent the curved shape.
        The default is 16.

    cellData : dict, optional
        dictionary containing cell centered data.
        Keys should be the names of the variable stored in each array.
        Arrays should have number of elements equal to
        ncells = npilars * nlayers.

    pointData : dict, optional
        dictionary containing node centered data.
        Keys should be the names of the variable stored in each array.
        Arrays should have number of elements equal to
        npoints = npilars * (nlayers + 1).

    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

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

    Returns
    -------

    str
        Full path to saved file.

    Notes
    -----

    This function only export vertical shapes for now.
    However, it should be easy to
    rotate the cylinder to represent other orientations.

    """
    # Define x, y coordinates from polar coordinates.
    dpi = 2.0 * np.pi / npilars
    angles = np.arange(0.0, 2.0 * np.pi, dpi)

    x = radius * np.cos(angles) + x0
    y = radius * np.sin(angles) + y0

    dz = (z1 - z0) / nlayers
    z = np.arange(z0, z1 + dz, step=dz)

    npoints = npilars * (nlayers + 1)
    ncells = npilars * nlayers

    xx = np.zeros(npoints)
    yy = np.zeros(npoints)
    zz = np.zeros(npoints)

    ii = 0
    for k in range(nlayers + 1):
        for p in range(npilars):
            xx[ii] = x[p]
            yy[ii] = y[p]
            zz[ii] = z[k]
            ii = ii + 1

    # Define connectivity
    conn = np.zeros(4 * ncells, dtype=np.int64)
    ii = 0
    for l in range(nlayers):
        for p in range(npilars):
            p0 = p
            if p + 1 == npilars:
                p1 = 0
            else:
                p1 = p + 1  # circular loop

            n0 = p0 + l * npilars
            n1 = p1 + l * npilars
            n2 = n0 + npilars
            n3 = n1 + npilars

            conn[ii + 0] = n0
            conn[ii + 1] = n1
            conn[ii + 2] = n3
            conn[ii + 3] = n2
            ii = ii + 4

    # Define offsets
    offsets = np.zeros(ncells, dtype=np.int64)
    for i in range(ncells):
        offsets[i] = (i + 1) * 4

    # Define cell types
    ctype = np.ones(ncells) + VtkPixel.tid

    return unstructuredGridToVTK(
        path,
        xx,
        yy,
        zz,
        connectivity=conn,
        offsets=offsets,
        cell_types=ctype,
        cellData=cellData,
        pointData=pointData,
        fieldData=fieldData,
        direct_format=direct_format,
        appended_format=appended_format,
        compression=compression,
        compressor=compressor,
        append=append,
    )
