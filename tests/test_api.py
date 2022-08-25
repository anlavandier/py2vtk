import os

import numpy as np
import pytest
from utils_test import get_vtk_data
from vtk import (
    vtkXMLImageDataReader,
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
    polyLinesToVTK,
    unstructuredGridToVTK,
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
