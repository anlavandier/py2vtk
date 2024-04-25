import os

import numpy as np
import pytest
from mpi4py import MPI
from utils_test import get_vtk_data
from vtkmodules.vtkIOXML import (
    vtkXMLPImageDataReader,
    vtkXMLPPolyDataReader,
    vtkXMLPRectilinearGridReader,
    vtkXMLPStructuredGridReader,
    vtkXMLPUnstructuredGridReader,
)

from py2vtk.core.vtkcells import VtkTetra, VtkTriangle, VtkVertex
from py2vtk.mpi.api import (
    parallelImageToVTK,
    parallelPolyDataToVTK,
    parallelRectilinearGridToVTK,
    parallelStructuredGridToVTK,
    parallelUnstructuredGridToVTK,
)

comm = MPI.COMM_WORLD.Dup()

# Tolerance for float equality
ATOL = 1e-15
RTOL = 1e-15


@pytest.mark.parallel
def test_parallelImageToVTK():
    rank = comm.Get_rank()
    size = comm.Get_size()

    path = "test_parallelImageToVTK"

    origin = (0, 0, 0)
    spacing = (1.0, 1.0, 1.0)

    start = (0, 0, 5 * rank)
    end = (5, 5, 5 * (rank + 1))

    # pointdata should be continous across processes as the boundary points
    # are repeated
    pointdata = np.arange(6 * 6 * (5 * size + 1))[
        6**2 * 5 * rank : 6**2 * (5 * (rank + 1) + 1)
    ]

    celldata = np.arange(5**3 * rank, 5**3 * (rank + 1)).reshape(5, 5, 5, order="F")

    parallelImageToVTK(
        path,
        starts=start,
        ends=end,
        origin=origin,
        spacing=spacing,
        cellData={"cell_number": celldata},
        pointData={"point_number": pointdata},
        fieldData=None,
        comm=comm,
        ghostlevel=0,
        direct_format="ascii",
        append=False,
    )

    if rank == 0:

        reader = vtkXMLPImageDataReader()

        vtk_point, vtk_cell, _ = get_vtk_data(
            reader, path + ".pvti", point="point_number", cell="cell_number", field=None
        )

        expected_cell = np.arange(5**3 * size)
        expected_point = np.arange(6 * 6 * (5 * size + 1))

        assert np.allclose(vtk_cell, expected_cell, atol=ATOL, rtol=RTOL)
        assert np.allclose(vtk_point, expected_point, atol=ATOL, rtol=RTOL)

        for r in range(size):
            os.remove(path + f".{r}.vti")
        os.remove(path + ".pvti")


@pytest.mark.parallel
def test_parallelRectilinearGridToVTK():
    rank = comm.Get_rank()
    size = comm.Get_size()

    path = "test_parallelRectilinearGridToVTK"

    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    z = np.linspace(rank, rank + 1, 6, endpoint=True)

    start = (0, 0, 5 * rank)
    end = (5, 5, 5 * (rank + 1))

    # pointdata should be continous across processes as the boundary points
    # are repeated
    pointdata = np.arange(6 * 6 * (5 * size + 1))[
        6**2 * 5 * rank : 6**2 * (5 * (rank + 1) + 1)
    ]

    celldata = np.arange(5**3 * rank, 5**3 * (rank + 1)).reshape(5, 5, 5, order="F")

    parallelRectilinearGridToVTK(
        path,
        x,
        y,
        z,
        starts=start,
        ends=end,
        cellData={"cell_number": celldata},
        pointData={"point_number": pointdata},
        fieldData=None,
        comm=comm,
        ghostlevel=0,
        direct_format="ascii",
        append=False,
    )

    if rank == 0:

        reader = vtkXMLPRectilinearGridReader()

        vtk_point, vtk_cell, _ = get_vtk_data(
            reader, path + ".pvtr", point="point_number", cell="cell_number", field=None
        )

        expected_cell = np.arange(5**3 * size)
        expected_point = np.arange(6 * 6 * (5 * size + 1))

        assert np.allclose(vtk_cell, expected_cell, atol=ATOL, rtol=RTOL)
        assert np.allclose(vtk_point, expected_point, atol=ATOL, rtol=RTOL)

        for r in range(size):
            os.remove(path + f".{r}.vtr")
        os.remove(path + ".pvtr")


@pytest.mark.parallel
def test_parallelStructuredGridToVTK():
    rank = comm.Get_rank()
    size = comm.Get_size()

    path = "test_parallelStructuredGridToVTK"

    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    z = np.linspace(rank, rank + 1, 6, endpoint=True)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    start = (0, 0, 5 * rank)
    end = (5, 5, 5 * (rank + 1))

    # pointdata should be continous across processes as the boundary points
    # are repeated
    pointdata = np.arange(6 * 6 * (5 * size + 1))[
        6**2 * 5 * rank : 6**2 * (5 * (rank + 1) + 1)
    ]

    celldata = np.arange(5**3 * rank, 5**3 * (rank + 1)).reshape(5, 5, 5, order="F")

    parallelStructuredGridToVTK(
        path,
        xx,
        yy,
        zz,
        starts=start,
        ends=end,
        cellData={"cell_number": celldata},
        pointData={"point_number": pointdata},
        fieldData=None,
        comm=comm,
        ghostlevel=0,
        direct_format="ascii",
        append=False,
    )
    comm.Barrier()
    if rank == 0:

        reader = vtkXMLPStructuredGridReader()

        vtk_point, vtk_cell, _ = get_vtk_data(
            reader, path + ".pvts", point="point_number", cell="cell_number", field=None
        )

        expected_cell = np.arange(5**3 * size)
        expected_point = np.arange(6 * 6 * (5 * size + 1))

        assert np.allclose(vtk_cell, expected_cell, atol=ATOL, rtol=RTOL)
        assert np.allclose(vtk_point, expected_point, atol=ATOL, rtol=RTOL)

        for r in range(size):
            os.remove(path + f".{r}.vts")
        os.remove(path + ".pvts")


@pytest.mark.parallel
def test_parallelPolyDataToVTK():
    rank = comm.Get_rank()
    size = comm.Get_size()

    path = "test_parallelPolyDataToVTK"

    x = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    z = np.array([0, 1, 0, 1, 0, 1, 0, 1]) + 2 * rank

    verts = np.array([0, 1, 2, 3])
    lines = (np.array([0, 1, 1, 2, 2, 3]), np.array([2, 4, 6]))
    strips = (np.array([3, 5, 6]), np.array([3]))
    polys = (np.array([0, 1, 3, 2, 4, 5, 7, 6]), np.array([8]))

    cell_v = np.array([0, 1, 2, 3]) + 9 * rank
    cell_l = np.array([4, 5, 6]) + 9 * rank
    cell_s = np.array([8]) + 9 * rank
    cell_p = np.array([7]) + 9 * rank

    celldata = np.concatenate([cell_v, cell_l, cell_p, cell_s])
    pointdata = np.arange(8 * rank, 8 * (rank + 1))

    parallelPolyDataToVTK(
        path,
        x,
        y,
        z,
        vertices=verts,
        lines=lines,
        strips=strips,
        polys=polys,
        cellData={"cell_number": celldata},
        pointData={"point_number": pointdata},
        fieldData=None,
        comm=comm,
        ghostlevel=0,
        direct_format="ascii",
        append=False,
    )
    comm.Barrier()

    if rank == 0:

        reader = vtkXMLPPolyDataReader()

        vtk_point, vtk_cell, _ = get_vtk_data(
            reader, path + ".pvtp", point="point_number", cell="cell_number", field=None
        )

        expected_cell = np.concatenate(
            [
                np.concatenate(
                    # Vertices
                    [np.array([0, 1, 2, 3]) + 9 * r for r in range(size)]
                ),
                np.concatenate(
                    # Lines
                    [np.array([4, 5, 6]) + 9 * r for r in range(size)]
                ),
                # Polys
                np.concatenate([np.array([7]) + 9 * r for r in range(size)]),
                # Strips
                np.concatenate([np.array([8]) + 9 * r for r in range(size)]),
            ]
        )
        expected_point = np.arange(8 * size)

        assert np.allclose(vtk_cell, expected_cell, atol=ATOL, rtol=RTOL)
        assert np.allclose(vtk_point, expected_point, atol=ATOL, rtol=RTOL)

        for r in range(size):
            os.remove(path + f".{r}.vtp")
        os.remove(path + ".pvtp")


@pytest.mark.parallel
def test_parallelUnstructuredGridToVTK():
    rank = comm.Get_rank()
    size = comm.Get_size()

    path = "test_parallelUnstructuredGridToVTK"

    x = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    z = np.array([0, 1, 0, 1, 0, 1, 0, 1]) + 2 * rank

    conn = np.array([0, 1, 0, 1, 2, 4, 5, 7, 6])
    offsets = np.array([1, 2, 5, 9])
    cell_types = np.array([VtkVertex.tid, VtkVertex.tid, VtkTriangle.tid, VtkTetra.tid])
    celldata = np.arange(4 * rank, 4 * (rank + 1))
    pointdata = np.arange(8 * rank, 8 * (rank + 1))

    parallelUnstructuredGridToVTK(
        path,
        x,
        y,
        z,
        connectivity=conn,
        offsets=offsets,
        cell_types=cell_types,
        cellData={"cell_number": celldata},
        pointData={"point_number": pointdata},
        fieldData=None,
        comm=comm,
        ghostlevel=0,
        direct_format="ascii",
        append=False,
    )
    comm.Barrier()
    if rank == 0:

        reader = vtkXMLPUnstructuredGridReader()

        vtk_point, vtk_cell, _ = get_vtk_data(
            reader, path + ".pvtu", point="point_number", cell="cell_number", field=None
        )

        expected_cell = np.arange(4 * size)
        expected_point = np.arange(8 * size)

        assert np.allclose(vtk_cell, expected_cell, atol=ATOL, rtol=RTOL)
        assert np.allclose(vtk_point, expected_point, atol=ATOL, rtol=RTOL)

        for r in range(size):
            os.remove(path + f".{r}.vtu")
        os.remove(path + ".pvtu")
