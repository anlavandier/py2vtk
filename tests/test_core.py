import os

import numpy as np
import pytest
from utils_test import get_vtk_data
from vtkmodules.vtkIOXML import (
    vtkXMLImageDataReader,
    vtkXMLPolyDataReader,
    vtkXMLRectilinearGridReader,
    vtkXMLStructuredGridReader,
    vtkXMLUnstructuredGridReader,
)

from py2vtk.core.vtkcells import VtkLine, VtkPolyLine, VtkTriangle, VtkVertex
from py2vtk.core.vtkfiles import (
    VtkFile,
    VtkImageData,
    VtkPolyData,
    VtkRectilinearGrid,
    VtkStructuredGrid,
    VtkUnstructuredGrid,
)

# Tolerance for float equality
ATOL = 1e-15
RTOL = 1e-15


@pytest.mark.serial
@pytest.mark.parametrize("compressor", ["zlib", "lzma"])
@pytest.mark.parametrize("compression", [True, False, 5])
@pytest.mark.parametrize("direct_format", ["binary", "ascii"])
@pytest.mark.parametrize("appended_format", ["binary", "raw"])
def test_image_data(compressor, compression, direct_format, appended_format):
    origin = 0, 3, 6
    spacing = 1.0, 2.0, 4.0

    filepath = "image_data"

    pointdata = np.random.random((6, 11, 8))
    celldata = np.random.random((5, 10, 7))

    if compression is not False and appended_format == "raw":
        with pytest.warns(UserWarning):
            vtk_file = VtkFile(
                filepath=filepath,
                ftype=VtkImageData,
                direct_format=direct_format,
                appended_format=appended_format,
                compression=compression,
                compressor=compressor,
            )
    else:
        vtk_file = VtkFile(
            filepath=filepath,
            ftype=VtkImageData,
            direct_format=direct_format,
            appended_format=appended_format,
            compression=compression,
            compressor=compressor,
        )

    vtk_file.openGrid(start=(0, 0, 0), end=(5, 10, 7), origin=origin, spacing=spacing)
    vtk_file.openPiece(start=(0, 0, 0), end=(5, 10, 7))

    vtk_file.openData("Point", scalars="points")
    vtk_file.addData("points", pointdata, append=True)
    vtk_file.closeData("Point")

    vtk_file.openData("Cell", scalars="cells")
    vtk_file.addData("cells", celldata, append=False)
    vtk_file.closeData("Cell")

    vtk_file.closePiece()

    vtk_file.openData("Field")
    vtk_file.addData("field", np.array([0]), append=False)
    vtk_file.closeData("Field")

    vtk_file.closeGrid()
    vtk_file.save()

    reader = vtkXMLImageDataReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, vtk_file.getFileName())

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")
    vtk_cell = vtk_cell.reshape(celldata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, celldata, atol=ATOL, rtol=RTOL)

    os.remove(vtk_file.getFileName())


@pytest.mark.serial
@pytest.mark.parametrize("compressor", ["zlib", "lzma"])
@pytest.mark.parametrize("compression", [True, False, 5])
@pytest.mark.parametrize("direct_format", ["binary", "ascii"])
@pytest.mark.parametrize("appended_format", ["binary", "raw"])
def test_rectilinear_grid(compressor, compression, direct_format, appended_format):
    filepath = "rectilinear_grid"

    x = np.random.random(6)
    y = np.random.random(11)
    z = np.random.random(8)
    pointdata = np.random.random((6, 11, 8))
    celldata = np.random.random((5, 10, 7))

    if compression is not False and appended_format == "raw":
        with pytest.warns(UserWarning):
            vtk_file = VtkFile(
                filepath=filepath,
                ftype=VtkRectilinearGrid,
                direct_format=direct_format,
                appended_format=appended_format,
                compression=compression,
                compressor=compressor,
            )
    else:
        vtk_file = VtkFile(
            filepath=filepath,
            ftype=VtkRectilinearGrid,
            direct_format=direct_format,
            appended_format=appended_format,
            compression=compression,
            compressor=compressor,
        )

    vtk_file.openGrid(start=(0, 0, 0), end=(5, 10, 7))
    vtk_file.openPiece(start=(0, 0, 0), end=(5, 10, 7))

    vtk_file.openData("Point", scalars="points")
    vtk_file.addData("points", pointdata, append=True)
    vtk_file.closeData("Point")

    vtk_file.openData("Cell", scalars="cells")
    vtk_file.addData("cells", celldata, append=False)
    vtk_file.closeData("Cell")

    vtk_file.openElement("Coordinates")
    vtk_file.addData("x_coordinates", x, append=True)
    vtk_file.addData("y_coordinates", y, append=False)
    vtk_file.addData("z_coordinates", z, append=True)
    vtk_file.closeElement("Coordinates")

    vtk_file.closePiece()

    vtk_file.openData("Field")
    vtk_file.addData("field", np.array([0]), append=False)
    vtk_file.closeData("Field")

    vtk_file.closeGrid()
    vtk_file.save()

    reader = vtkXMLRectilinearGridReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, vtk_file.getFileName())

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")
    vtk_cell = vtk_cell.reshape(celldata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, celldata, atol=ATOL, rtol=RTOL)

    os.remove(vtk_file.getFileName())


@pytest.mark.serial
@pytest.mark.parametrize("compressor", ["zlib", "lzma"])
@pytest.mark.parametrize("compression", [True, False, 5])
@pytest.mark.parametrize("direct_format", ["binary", "ascii"])
@pytest.mark.parametrize("appended_format", ["binary", "raw"])
def test_structured_grid(compressor, compression, direct_format, appended_format):
    filepath = "structured_grid"

    x = np.random.random((6, 11, 8))
    y = np.random.random((6, 11, 8))
    z = np.random.random((6, 11, 8))

    pointdata = np.random.random((6, 11, 8))
    celldata = np.random.random((5, 10, 7))

    if compression is not False and appended_format == "raw":
        with pytest.warns(UserWarning):
            vtk_file = VtkFile(
                filepath=filepath,
                ftype=VtkStructuredGrid,
                direct_format=direct_format,
                appended_format=appended_format,
                compression=compression,
                compressor=compressor,
            )
    else:
        vtk_file = VtkFile(
            filepath=filepath,
            ftype=VtkStructuredGrid,
            direct_format=direct_format,
            appended_format=appended_format,
            compression=compression,
            compressor=compressor,
        )

    vtk_file.openGrid(start=(0, 0, 0), end=(5, 10, 7))
    vtk_file.openPiece(start=(0, 0, 0), end=(5, 10, 7))

    vtk_file.openData("Point", scalars="points")
    vtk_file.addData("points", pointdata, append=False)
    vtk_file.closeData("Point")

    vtk_file.openData("Cell", scalars="cells")
    vtk_file.addData("cells", celldata, append=False)
    vtk_file.closeData("Cell")

    vtk_file.openElement("Points")
    vtk_file.addData("points", (x, y, z), append=True)
    vtk_file.closeElement("Points")

    vtk_file.closePiece()

    vtk_file.openData("Field")
    vtk_file.addData("field", np.array([0]), append=False)
    vtk_file.closeData("Field")

    vtk_file.closeGrid()
    vtk_file.save()

    reader = vtkXMLStructuredGridReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, vtk_file.getFileName())

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")
    vtk_cell = vtk_cell.reshape(celldata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, celldata, atol=ATOL, rtol=RTOL)

    os.remove(vtk_file.getFileName())


@pytest.mark.serial
@pytest.mark.parametrize("compressor", ["zlib", "lzma"])
@pytest.mark.parametrize("compression", [True, False, 5])
@pytest.mark.parametrize("direct_format", ["binary", "ascii"])
@pytest.mark.parametrize("appended_format", ["binary", "raw"])
def test_poly_data(compressor, compression, direct_format, appended_format):
    filepath = "poly_data"

    x = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    y = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
    z = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 1])

    verts_conn = np.array([0])
    verts_off = np.array([1])

    lines_conn = np.array([1, 2])
    lines_off = np.array([2])

    strips_conn = np.array([3, 4, 5])
    strips_off = np.array([3])

    polys_conn = np.array([6, 7, 8, 9])
    polys_off = np.array([4])

    pointdata = np.random.random(size=10)

    fielddata = np.array(0)

    if compression is not False and appended_format == "raw":
        with pytest.warns(UserWarning):
            vtk_file = VtkFile(
                filepath,
                VtkPolyData,
                direct_format=direct_format,
                appended_format=appended_format,
                compression=compression,
                compressor=compressor,
            )
    else:
        vtk_file = VtkFile(
            filepath,
            VtkPolyData,
            direct_format=direct_format,
            appended_format=appended_format,
            compression=compression,
            compressor=compressor,
        )

    vtk_file.openGrid()
    vtk_file.openPiece(npoints=10, nverts=1, nlines=1, nstrips=1, npolys=1)

    vtk_file.openElement("Points")
    vtk_file.addData("points", (x, y, z), append=True)
    vtk_file.closeElement("Points")

    vtk_file.openElement("Verts")
    vtk_file.addData("connectivity", data=verts_conn, append=False)
    vtk_file.addData("offsets", data=verts_off, append=False)
    vtk_file.closeElement("Verts")

    vtk_file.openElement("Lines")
    vtk_file.addData("connectivity", data=lines_conn, append=True)
    vtk_file.addData("offsets", data=lines_off, append=True)
    vtk_file.closeElement("Lines")

    vtk_file.openElement("Strips")
    vtk_file.addData("connectivity", data=strips_conn, append=False)
    vtk_file.addData("offsets", data=strips_off, append=False)
    vtk_file.closeElement("Strips")

    vtk_file.openElement("Polys")
    vtk_file.addData("connectivity", data=polys_conn, append=True)
    vtk_file.addData("offsets", data=polys_off, append=True)
    vtk_file.closeElement("Polys")

    vtk_file.openData("Point", scalars="points")
    vtk_file.addData("points", pointdata, append=False)
    vtk_file.closeData("Point")

    vtk_file.closePiece()

    vtk_file.openData("Field")
    vtk_file.addData("field", fielddata, append=False)
    vtk_file.closeData("Field")

    vtk_file.closeGrid()
    vtk_file.save()

    reader = vtkXMLPolyDataReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(
        reader, vtk_file.getFileName(), cell=None
    )

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)

    os.remove(vtk_file.getFileName())


@pytest.mark.serial
@pytest.mark.parametrize("compressor", ["zlib", "lzma"])
@pytest.mark.parametrize("compression", [True, False, 5])
@pytest.mark.parametrize("direct_format", ["binary", "ascii"])
@pytest.mark.parametrize("appended_format", ["binary", "raw"])
def test_unstructured_grid(compressor, compression, direct_format, appended_format):
    filepath = "unstructured_grid"

    x = np.array([0, 0, 0])
    y = np.array([0, 1, 2])
    z = np.array([0.5, 2, 3])

    connectivity = np.array([0, 1, 2, 1, 2, 1, 0, 2, 1])
    offsets = np.array([3, 5, 6, 9])
    celltypes = np.array([VtkTriangle.tid, VtkLine.tid, VtkVertex.tid, VtkPolyLine.tid])
    pointdata = np.random.random(3)
    celldata = np.random.random(4)

    if compression is not False and appended_format == "raw":
        with pytest.warns(UserWarning):
            vtk_file = VtkFile(
                filepath=filepath,
                ftype=VtkUnstructuredGrid,
                direct_format=direct_format,
                appended_format=appended_format,
                compression=compression,
                compressor=compressor,
            )
    else:
        vtk_file = VtkFile(
            filepath=filepath,
            ftype=VtkUnstructuredGrid,
            direct_format=direct_format,
            appended_format=appended_format,
            compression=compression,
            compressor=compressor,
        )

    vtk_file.openGrid()
    vtk_file.openPiece(ncells=4, npoints=3)

    vtk_file.openData("Point", scalars="points")
    vtk_file.addData("points", pointdata, append=False)
    vtk_file.closeData("Point")

    vtk_file.openData("Cell", scalars="cells")
    vtk_file.addData("cells", celldata, append=False)
    vtk_file.closeData("Cell")

    vtk_file.openElement("Points")
    vtk_file.addData("points", (x, y, z), append=True)
    vtk_file.closeElement("Points")

    vtk_file.openElement("Cells")
    vtk_file.addData("connectivity", connectivity, append=False)
    vtk_file.addData("offsets", offsets, append=False)
    vtk_file.addData("types", celltypes, append=False)
    vtk_file.closeElement("Cells")

    vtk_file.closePiece()

    vtk_file.openData("Field")
    vtk_file.addData("field", np.array([0]), append=False)
    vtk_file.closeData("Field")

    vtk_file.closeGrid()
    vtk_file.save()

    reader = vtkXMLUnstructuredGridReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, vtk_file.getFileName())

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")
    vtk_cell = vtk_cell.reshape(celldata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, celldata, atol=ATOL, rtol=RTOL)

    os.remove(vtk_file.getFileName())
