import os

import numpy as np
import pytest
from utils_test import get_vtk_data
from vtkmodules.vtkIOXML import (
    vtkXMLImageDataReader,
    vtkXMLPImageDataReader,
    vtkXMLPolyDataReader,
    vtkXMLPPolyDataReader,
    vtkXMLPRectilinearGridReader,
    vtkXMLPStructuredGridReader,
    vtkXMLPUnstructuredGridReader,
    vtkXMLRectilinearGridReader,
    vtkXMLStructuredGridReader,
    vtkXMLUnstructuredGridReader,
)

from py2vtk.api import (
    cylinderToVTK,
    gridToVTK,
    imageToVTK,
    linesToVTK,
    pointsToVTK,
    polyDataToVTK,
    polyLinesToVTK,
    unstructuredGridToVTK,
)
from py2vtk.api.parallel import (
    writeParallelVTKGrid,
    writeParallelVTKImageData,
    writeParallelVTKPolyData,
    writeParallelVTKUnstructuredGrid,
)
from py2vtk.core.vtkcells import VtkLine, VtkPolyLine, VtkTriangle, VtkVertex

# Tolerance for float equality
ATOL = 1e-15
RTOL = 1e-15


@pytest.mark.serial
@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("origin", [(0, 0, 0), (-1, 2, 4), (-5, 2, 3)])
@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (1.0, 2.0, 3.0)])
@pytest.mark.parametrize("data_size", [(5, 5, 5), (2, 3, 5)])
def test_imageToVTK(append, origin, spacing, data_size):

    pointdata = np.random.random(size=data_size)
    celldata = np.random.random(size=tuple(np.array(data_size) - 1))

    filename = imageToVTK(
        "test_imageToVTK",
        start=(0, 0, 0),
        end=None,
        origin=origin,
        spacing=spacing,
        cellData={"cells": celldata},
        pointData={"points": pointdata},
        fieldData={"field": np.array([0])},
        append=append,
    )

    reader = vtkXMLImageDataReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, filename)

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")
    vtk_cell = vtk_cell.reshape(celldata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, celldata, atol=ATOL, rtol=RTOL)

    os.remove(filename)


@pytest.mark.serial
@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("data_size", [(5, 5, 5), (2, 3, 5)])
@pytest.mark.parametrize("mode", ["rectilinear", "structured"])
def test_gridToVTK(
    append,
    data_size,
    mode,
):
    x = np.linspace(-1, 2, data_size[0])
    y = np.linspace(3, 9, data_size[1])
    z = np.linspace(0, 2, data_size[2])

    if mode == "structured":
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

    pointdata = np.random.random(size=data_size)
    celldata = np.random.random(size=tuple(np.array(data_size) - 1))

    filename = gridToVTK(
        f"test_{mode}GridToVTK",
        x,
        y,
        z,
        start=(0, 0, 0),
        cellData={"cells": celldata},
        pointData={"points": pointdata},
        fieldData={"field": np.array([0])},
        append=append,
    )

    if mode == "structured":
        reader = vtkXMLStructuredGridReader()
    else:
        reader = vtkXMLRectilinearGridReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, filename)

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")
    vtk_cell = vtk_cell.reshape(celldata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, celldata, atol=ATOL, rtol=RTOL)

    os.remove(filename)


@pytest.mark.serial
@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("nverts", [0, 5])
@pytest.mark.parametrize("nlines", [0, 5])
@pytest.mark.parametrize("nstrips", [0, 5])
@pytest.mark.parametrize("npolys", [0, 5])
def test_polyDataToVTK(
    append,
    nverts,
    nlines,
    nstrips,
    npolys,
):
    x = np.linspace(-1, 2, 20)
    y = np.linspace(3, 9, 20)
    z = np.linspace(0, 2, 20)

    verts = np.random.randint(low=0, high=20, size=nverts)
    if nlines == 0:
        lines = None
    else:
        lines = [
            np.random.randint(low=0, high=20, size=np.random.randint(low=2, high=4))
            for _ in range(nlines)
        ]

    if nstrips == 0:
        strips = None
    else:
        strips = [
            np.random.randint(low=0, high=20, size=np.random.randint(low=3, high=5))
            for _ in range(nstrips)
        ]

    if npolys == 0:
        polys = None
    else:
        polys = [
            np.random.randint(low=0, high=20, size=np.random.randint(low=4, high=7))
            for _ in range(npolys)
        ]

    pointdata = np.random.random(size=20)
    celldata = np.random.random(size=nverts + nlines + nstrips + npolys)

    filename = polyDataToVTK(
        "test_polyDataToVTK",
        x,
        y,
        z,
        vertices=verts,
        lines=lines,
        strips=strips,
        polys=polys,
        cellData={"cells": celldata},
        pointData={"points": pointdata},
        fieldData={"field": np.array([0])},
        append=append,
    )

    reader = vtkXMLPolyDataReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, filename)

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")
    vtk_cell = vtk_cell.reshape(celldata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, celldata, atol=ATOL, rtol=RTOL)

    os.remove(filename)


@pytest.mark.serial
@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("data_size", [(5, 5, 5), (2, 3, 5)])
def test_pointsToVTK(
    append,
    data_size,
):
    x = np.linspace(-1, 2, data_size[0])
    y = np.linspace(3, 9, data_size[1])
    z = np.linspace(0, 2, data_size[2])

    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    pointdata = np.random.random(size=data_size)

    filename = pointsToVTK(
        "test_pointsToVTK",
        x,
        y,
        z,
        data={"points": pointdata},
        fieldData={"field": np.array([0])},
        append=append,
    )

    reader = vtkXMLUnstructuredGridReader()

    vtk_point, _, vtk_field = get_vtk_data(reader, filename, cell=None)

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)

    os.remove(filename)


@pytest.mark.serial
@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("data_size", [20, 256])
def test_linesToVTK(
    append,
    data_size,
):
    x = np.linspace(-1, 2, data_size)
    y = np.linspace(3, 9, data_size)
    z = np.linspace(0, 2, data_size)

    pointdata = np.random.random(size=data_size)
    celldata = np.random.random(size=data_size // 2)

    filename = linesToVTK(
        "test_linesToVTK",
        x,
        y,
        z,
        pointData={"points": pointdata},
        cellData={"cells": celldata},
        fieldData={"field": np.array([0])},
        append=append,
    )

    reader = vtkXMLUnstructuredGridReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, filename)

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")
    vtk_cell = vtk_cell.reshape(celldata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, celldata, atol=ATOL, rtol=RTOL)

    os.remove(filename)


@pytest.mark.serial
@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("data_size", [20, 256])
def test_polyLinesToVTK(
    append,
    data_size,
):
    x = np.linspace(-1, 2, data_size)
    y = np.linspace(3, 9, data_size)
    z = np.linspace(0, 2, data_size)

    pointsperline = np.random.randint(
        low=2, high=data_size // 2 + 1, size=data_size // 2
    )
    print(data_size // 2, pointsperline.shape)
    sum = 0
    for i in range(data_size // 2):
        sum += pointsperline[i]
        if sum > data_size:
            if data_size - (sum - pointsperline[i]) == 1:
                pointsperline = pointsperline[:i]
                pointsperline[-1] += 1
            else:
                pointsperline = pointsperline[: i + 1]
                pointsperline[-1] = data_size - (sum - pointsperline[i])

            break

    assert np.sum(pointsperline) == data_size

    pointdata = np.random.random(size=data_size)
    celldata = np.random.random(size=pointsperline.size)

    filename = polyLinesToVTK(
        "test_polyLinesToVTK",
        x,
        y,
        z,
        pointsPerLine=pointsperline,
        pointData={"points": pointdata},
        cellData={"cells": celldata},
        fieldData={"field": np.array([0])},
        append=append,
    )

    reader = vtkXMLUnstructuredGridReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, filename)

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")
    vtk_cell = vtk_cell.reshape(celldata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, celldata, atol=ATOL, rtol=RTOL)

    os.remove(filename)


@pytest.mark.serial
@pytest.mark.parametrize("append", [True, False])
def test_unstructuredGridToVTK(
    append,
):
    x = np.array([0, 0, 0])
    y = np.array([0, 1, 2])
    z = np.array([0.5, 2, 3])

    connectivity = np.array([0, 1, 2, 1, 2, 1, 0, 2, 1])
    offsets = np.array([3, 5, 6, 9])
    celltypes = np.array([VtkTriangle.tid, VtkLine.tid, VtkVertex.tid, VtkPolyLine.tid])
    pointdata = np.random.random(3)
    celldata = np.random.random(4)

    filename = unstructuredGridToVTK(
        "test_unstructuredGridToVTK",
        x,
        y,
        z,
        connectivity=connectivity,
        offsets=offsets,
        cell_types=celltypes,
        pointData={"points": pointdata},
        cellData={"cells": celldata},
        fieldData={"field": np.array(0)},
        append=append,
    )

    reader = vtkXMLUnstructuredGridReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, filename)

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")
    vtk_cell = vtk_cell.reshape(celldata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, celldata, atol=ATOL, rtol=RTOL)

    os.remove(filename)


@pytest.mark.serial
@pytest.mark.parametrize("append", [True, False])
def test_cylinderToVTK(
    append,
):
    center = (0, 0)
    x_0, y_0 = center

    z_bounds = (-1, 1)
    z_0, z_1 = z_bounds

    radius = 3

    nlayers = 10
    npilars = 15

    pointdata = np.random.random(npilars * (nlayers + 1))
    celldata = np.random.random(npilars * nlayers)

    filename = cylinderToVTK(
        "test_cylinderToVTK",
        x0=x_0,
        y0=y_0,
        z0=z_0,
        z1=z_1,
        radius=radius,
        nlayers=nlayers,
        npilars=npilars,
        pointData={"points": pointdata},
        cellData={"cells": celldata},
        fieldData={"field": np.array(0)},
        append=append,
    )

    reader = vtkXMLUnstructuredGridReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, filename)

    vtk_point = vtk_point.reshape(pointdata.shape, order="F")
    vtk_cell = vtk_cell.reshape(celldata.shape, order="F")

    assert vtk_field == [0]
    assert np.allclose(vtk_point, pointdata, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, celldata, atol=ATOL, rtol=RTOL)

    os.remove(filename)


@pytest.mark.serial
def test_writeParallelVTKImageData():
    filepath = "test_writeParallelVTKImageData"

    origin = (0, 0, 0)
    spacing = (1.0, 1.0, 1.0)

    start_1, end_1 = (0, 0, 0), (5, 1, 1)
    start_2, end_2 = (5, 0, 0), (10, 1, 1)

    celldata1 = np.arange(0, 5).reshape(5, 1, 1)
    celldata2 = np.arange(5, 10).reshape(5, 1, 1)
    pointdata = np.random.random(size=(6, 2, 2))
    ghostlevel = 0

    fp1 = imageToVTK(
        filepath + "1",
        start=start_1,
        end=end_1,
        origin=origin,
        spacing=spacing,
        cellData={"cell_num": celldata1},
        pointData={"points": pointdata},
        append=False,
        fieldData={"field": np.array(0)},
    )

    fp2 = imageToVTK(
        filepath + "2",
        start=start_2,
        end=end_2,
        origin=origin,
        spacing=spacing,
        cellData={"cell_num": celldata2},
        pointData={"points": pointdata},
        append=False,
        fieldData={"field": np.array(0)},
    )

    celldata_info = {"cell_num": (np.dtype("int64"), 1)}
    pointdata_info = {"points": (np.dtype("float64"), 1)}

    filename = writeParallelVTKImageData(
        filepath,
        starts=[start_1, start_2],
        ends=[end_1, end_2],
        origin=origin,
        spacing=spacing,
        sources=[filepath + "1" + ".vti", filepath + "2" + ".vti"],
        ghostlevel=ghostlevel,
        cellData=celldata_info,
        pointData=pointdata_info,
        fieldData={"field": np.array(0)},
        format="ascii",
    )

    reader = vtkXMLPImageDataReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, filename, cell="cell_num")

    assert vtk_field == np.array(0)

    expected_point = np.concatenate([pointdata[:-1, :, :], pointdata], axis=0)
    expected_cell = np.concatenate([celldata1, celldata2], axis=0)

    vtk_point = vtk_point.reshape(expected_point.shape, order="F")
    vtk_cell = vtk_cell.reshape(expected_cell.shape, order="F")

    assert np.allclose(vtk_point, expected_point, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, expected_cell, atol=ATOL, rtol=RTOL)

    os.remove(fp1)
    os.remove(fp2)
    os.remove(filename)


@pytest.mark.serial
@pytest.mark.parametrize("mode", ["rectilinear", "structured"])
def test_writeParallelVTKGrid(mode):
    filepath = "test_writeParallelVTK" + mode + "Grid"

    start_1, end_1 = (0, 0, 0), (5, 3, 3)
    start_2, end_2 = (5, 0, 0), (10, 3, 3)
    x1 = np.linspace(-1, 2, 6)
    x2 = np.linspace(2, 4, 6)
    y = np.linspace(3, 9, 4)
    z = np.linspace(0, 2, 4)

    if mode == "rectilinear":
        y1, y2 = y, y
        z1, z2 = z, z
        ext = ".vtr"
    if mode == "structured":
        x1, y1, z1 = np.meshgrid(x1, y, z, indexing="ij")
        x2, y2, z2 = np.meshgrid(x2, y, z, indexing="ij")
        ext = ".vts"

    celldata1 = np.arange(0, 45).reshape(5, 3, 3)
    celldata2 = np.arange(45, 90).reshape(5, 3, 3)
    pointdata = np.random.random(size=(6, 4, 4))
    ghostlevel = 0

    fp1 = gridToVTK(
        filepath + "1",
        x1,
        y1,
        z1,
        start=start_1,
        cellData={"cell_num": celldata1},
        pointData={"points": pointdata},
        append=False,
        fieldData={"field": np.array(0)},
    )

    fp2 = gridToVTK(
        filepath + "2",
        x2,
        y2,
        z2,
        start=start_2,
        cellData={"cell_num": celldata2},
        pointData={"points": pointdata},
        append=False,
        fieldData={"field": np.array(0)},
    )

    celldata_info = {"cell_num": (np.dtype("int64"), 1)}
    pointdata_info = {"points": (np.dtype("float64"), 1)}

    filename = writeParallelVTKGrid(
        filepath,
        coordsDtype=np.dtype("float64"),
        starts=[start_1, start_2],
        ends=[end_1, end_2],
        sources=[filepath + "1" + ext, filepath + "2" + ext],
        ghostlevel=ghostlevel,
        cellData=celldata_info,
        pointData=pointdata_info,
        fieldData={"field": np.array(0)},
        format="ascii",
    )
    if mode == "rectilinear":
        reader = vtkXMLPRectilinearGridReader()
    if mode == "structured":
        reader = vtkXMLPStructuredGridReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, filename, cell="cell_num")

    assert vtk_field == np.array(0)

    expected_point = np.concatenate([pointdata[:-1, :, :], pointdata], axis=0)
    expected_cell = np.concatenate([celldata1, celldata2], axis=0)

    vtk_point = vtk_point.reshape(expected_point.shape, order="F")
    vtk_cell = vtk_cell.reshape(expected_cell.shape, order="F")

    assert np.allclose(vtk_point, expected_point, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, expected_cell, atol=ATOL, rtol=RTOL)

    os.remove(fp1)
    os.remove(fp2)
    os.remove(filename)


@pytest.mark.serial
@pytest.mark.parametrize("nverts", [0, 5])
@pytest.mark.parametrize("nlines", [0, 5])
@pytest.mark.parametrize("nstrips", [0, 5])
@pytest.mark.parametrize("npolys", [0, 5])
def test_writeParallelVTKPolyData(
    nverts,
    nlines,
    nstrips,
    npolys,
):
    filepath = "test_writeParallelVTKPolyData"

    x1 = np.random.random(size=20)
    x2 = np.random.random(size=20) + 1
    y = np.random.random(size=20)
    z = np.random.random(size=20)

    verts1 = np.random.randint(low=0, high=20, size=nverts // 2)
    verts2 = np.random.randint(low=0, high=20, size=nverts - nverts // 2)

    if nlines == 0:
        lines1, lines2 = None, None
    else:
        lines1 = [
            np.random.randint(low=0, high=20, size=np.random.randint(low=2, high=4))
            for _ in range(nlines // 2)
        ]
        lines2 = [
            np.random.randint(low=0, high=20, size=np.random.randint(low=2, high=4))
            for _ in range(nlines - nlines // 2)
        ]

    if nstrips == 0:
        strips1, strips2 = None, None
    else:
        strips1 = [
            np.random.randint(low=0, high=20, size=np.random.randint(low=3, high=5))
            for _ in range(nstrips // 2)
        ]
        strips2 = [
            np.random.randint(low=0, high=20, size=np.random.randint(low=3, high=5))
            for _ in range(nstrips - nstrips // 2)
        ]

    if npolys == 0:
        polys1, polys2 = None, None
    else:
        polys1 = [
            np.random.randint(low=0, high=20, size=np.random.randint(low=4, high=7))
            for _ in range(npolys // 2)
        ]
        polys2 = [
            np.random.randint(low=0, high=20, size=np.random.randint(low=4, high=7))
            for _ in range(npolys - npolys // 2)
        ]

    pointdata = np.random.random(size=20)
    cell_v1 = [0, 1][:nverts]
    cell_v2 = [2, 3, 4][:nverts]
    cell_l1 = [5, 6][:nlines]
    cell_l2 = [7, 8, 9][:nlines]
    cell_s1 = [10, 11][:nstrips]
    cell_s2 = [12, 13, 14][:nstrips]
    cell_p1 = [15, 16][:npolys]
    cell_p2 = [17, 18, 19][:npolys]

    celldata1 = np.concatenate([cell_v1, cell_l1, cell_s1, cell_p1]).astype("int")
    celldata2 = np.concatenate([cell_v2, cell_l2, cell_s2, cell_p2]).astype("int")

    ghostlevel = 0

    fp1 = polyDataToVTK(
        filepath + "1",
        x1,
        y,
        z,
        vertices=verts1,
        lines=lines1,
        strips=strips1,
        polys=polys1,
        cellData={"cell_num": celldata1},
        pointData={"points": pointdata},
        fieldData={"field": np.array([0])},
        append=False,
    )

    fp2 = polyDataToVTK(
        filepath + "2",
        x2,
        y,
        z,
        vertices=verts2,
        lines=lines2,
        strips=strips2,
        polys=polys2,
        cellData={"cell_num": celldata2},
        pointData={"points": pointdata},
        fieldData={"field": np.array([0])},
        append=False,
    )

    celldata_info = {"cell_num": (np.dtype("int64"), 1)}
    pointdata_info = {"points": (np.dtype("float64"), 1)}

    filename = writeParallelVTKPolyData(
        filepath,
        coordsDtype=np.dtype("float64"),
        sources=[filepath + "1" + ".vtp", filepath + "2" + ".vtp"],
        ghostlevel=ghostlevel,
        cellData=celldata_info,
        pointData=pointdata_info,
        fieldData={"field": np.array(0)},
        format="ascii",
    )
    reader = vtkXMLPPolyDataReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, filename, cell="cell_num")

    assert vtk_field == np.array(0)

    expected_point = np.concatenate([pointdata, pointdata], axis=0)
    expected_cell = np.concatenate(
        [cell_v1, cell_v2, cell_l1, cell_l2, cell_s1, cell_s2, cell_p1, cell_p2], axis=0
    )

    vtk_point = vtk_point.reshape(expected_point.shape, order="F")
    vtk_cell = vtk_cell.reshape(expected_cell.shape, order="F")

    assert np.allclose(vtk_point, expected_point, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, expected_cell, atol=ATOL, rtol=RTOL)

    os.remove(fp1)
    os.remove(fp2)
    os.remove(filename)


@pytest.mark.serial
def test_writeParallelVTKUnstructuredGrid():
    filepath = "test_writeParallelVTKUnstructuredGrid"

    ghostlevel = 0

    x1 = np.array([0, 0, 0])
    x2 = np.array([1, 1, 1])
    y = np.array([0, 1, 2])
    z = np.array([0.5, 2, 3])

    connectivity = np.array([0, 1, 2, 1, 2, 1, 0, 2, 1])
    offsets = np.array([3, 5, 6, 9])
    celltypes = np.array([VtkTriangle.tid, VtkLine.tid, VtkVertex.tid, VtkPolyLine.tid])
    pointdata = np.random.random(3)
    celldata1 = np.arange(offsets.size)
    celldata2 = np.arange(offsets.size) + celldata1.size

    fp1 = unstructuredGridToVTK(
        filepath + "1",
        x1,
        y,
        z,
        connectivity=connectivity,
        offsets=offsets,
        cell_types=celltypes,
        pointData={"points": pointdata},
        cellData={"cell_num": celldata1},
        fieldData={"field": np.array(0)},
        append=False,
    )

    fp2 = unstructuredGridToVTK(
        filepath + "2",
        x2,
        y,
        z,
        connectivity=connectivity,
        offsets=offsets,
        cell_types=celltypes,
        pointData={"points": pointdata},
        cellData={"cell_num": celldata2},
        fieldData={"field": np.array(0)},
        append=False,
    )

    celldata_info = {"cell_num": (np.dtype("int64"), 1)}
    pointdata_info = {"points": (np.dtype("float64"), 1)}

    filename = writeParallelVTKUnstructuredGrid(
        filepath,
        coordsDtype=np.dtype("float64"),
        sources=[filepath + "1" + ".vtu", filepath + "2" + ".vtu"],
        ghostlevel=ghostlevel,
        cellData=celldata_info,
        pointData=pointdata_info,
        fieldData={"field": np.array(0)},
        format="ascii",
    )

    reader = vtkXMLPUnstructuredGridReader()

    vtk_point, vtk_cell, vtk_field = get_vtk_data(reader, filename, cell="cell_num")

    assert vtk_field == np.array(0)

    expected_point = np.concatenate([pointdata, pointdata], axis=0)
    expected_cell = np.concatenate([celldata1, celldata2], axis=0)

    vtk_point = vtk_point.reshape(expected_point.shape, order="F")
    vtk_cell = vtk_cell.reshape(expected_cell.shape, order="F")

    assert np.allclose(vtk_point, expected_point, atol=ATOL, rtol=RTOL)
    assert np.allclose(vtk_cell, expected_cell, atol=ATOL, rtol=RTOL)

    os.remove(fp1)
    os.remove(fp2)
    os.remove(filename)
