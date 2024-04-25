import numpy as np

from py2vtk.api.serial import unstructuredGridToVTK
from py2vtk.core.vtkcells import VtkPolyhedron

path = "polyhedra"

x = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1.5, 2, 2, 2, 2, 3, 3, 3, 3, 1.5])
y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0.5, 0, 0, 1, 1, 0, 0, 1, 1, 0.5])
z = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0.5, 0, 1, 0, 1, 0, 1, 0, 1, 0.5])


conn = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
offsets = np.array([9, 17])
faces = np.array(
    [
        9,  # Number of faces for the first polyhedron
        4,
        0,
        1,
        3,
        2,  # Face 1 preceded by its number of points
        4,
        0,
        1,
        5,
        4,
        4,
        0,
        2,
        6,
        4,
        4,
        7,
        5,
        1,
        3,
        4,
        7,
        6,
        2,
        3,
        3,
        7,
        8,
        6,
        3,
        6,
        8,
        4,
        3,
        4,
        8,
        5,
        3,
        5,
        8,
        7,
        9,  # Number of faces for the second polyhedron
        4,
        9,
        10,
        14,
        13,  # Face 1 preceded by its number of points
        4,
        9,
        11,
        15,
        13,
        4,
        16,
        14,
        10,
        12,
        4,
        16,
        15,
        13,
        14,
        4,
        16,
        15,
        11,
        12,
        3,
        9,
        17,
        10,
        3,
        9,
        17,
        11,
        3,
        10,
        17,
        12,
        3,
        12,
        17,
        11,
    ]
)
#                       1 + (1 + number of points 1st face type) * number of faces of first face type
#                         + (1 + number of points 2nd face type) * number of faces of second face type
faceoffsets = np.array([1 + (1 + 4) * 5 + (1 + 3) * 4, len(faces)])
cell_types = np.array([VtkPolyhedron.tid] * 2)
celldata = {"cell_num": np.array([1, 2])}
pointdata = {"points": np.random.random(size=18)}

unstructuredGridToVTK(
    path,
    x,
    y,
    z,
    conn,
    offsets,
    cell_types,
    faces=faces,
    faceoffsets=faceoffsets,
    cellData=celldata,
    pointData=pointdata,
    append=False,
)
