import numpy as np

from py2vtk import imageToVTK

# Dimensions
nx, ny, nz = 6, 6, 2
ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)

# Variables
pressure = np.random.rand(ncells).reshape((nx, ny, nz), order="C")
temp = np.random.rand(npoints).reshape((nx + 1, ny + 1, nz + 1))

imageToVTK("./image", cellData={"pressure": pressure}, pointData={"temp": temp})

fluxx = np.random.rand(ncells).reshape((nx, ny, nz), order="F")
fluxy = np.random.rand(ncells).reshape((nx, ny, nz), order="F")
fluxz = np.random.rand(ncells).reshape((nx, ny, nz), order="F")
flux = (fluxx, fluxy, fluxz)

Efieldx = np.random.rand(npoints).reshape((nx + 1, ny + 1, nz + 1), order="F")
Efieldy = np.random.rand(npoints).reshape((nx + 1, ny + 1, nz + 1), order="F")
Efieldz = np.random.rand(npoints).reshape((nx + 1, ny + 1, nz + 1), order="F")
Efield = (Efieldx, Efieldy, Efieldz)

imageToVTK(
    "./image", cellData={"flux": flux}, pointData={"Efield": Efieldx}, append=False
)
