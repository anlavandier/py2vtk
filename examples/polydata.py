import numpy as np

from py2vtk.api.serial import polyDataToVTK

x = np.array([0, 0, 0, 1, 1, 1])
y = np.array([0, 1, 0, 0, 1, 0])
z = np.array([0, 0, 1, 0, 0, 1])

verts = np.array([0, 1, 2])
lines = (np.array([0, 1, 4, 5]), np.array([2, 4]))
strips = (np.array([0, 1, 2, 3, 4, 5]), np.array([6]))
polys = (np.array([0, 1, 4, 3]), np.array([4]))

pointdata = {"point": np.random.random(size=6)}
celldata = {"cell_num": np.arange(3 + 2 + 1 + 1)}

polyDataToVTK(
    "./polydata",
    x,
    y,
    z,
    verts,
    lines,
    strips,
    polys,
    cellData=celldata,
    pointData=pointdata,
    append=False,
)
