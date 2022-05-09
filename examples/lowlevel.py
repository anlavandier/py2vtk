
# ***********************************************************************************
# * Copyright 2010 - 2016 Paulo A. Herrera. All rights reserved                     *
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ***********************************************************************************

# **************************************************************
# * Example of how to use the low level VtkFile class.         *
# **************************************************************
import sys
import os
base_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(base_dir, '../py2vtk')
sys.path.insert(0, base_dir)

from core.vtkfiles import VtkFile, VtkRectilinearGrid
import numpy as np

nx, ny, nz = 6, 6, 2
lx, ly, lz = 1.0, 1.0, 1.0
dx, dy, dz = lx / nx, ly / ny, lz / nz
ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)
x = np.arange(0, lx + 0.1 * dx, dx, dtype="float64")
y = np.arange(0, ly + 0.1 * dy, dy, dtype="float64")
z = np.arange(0, lz + 0.1 * dz, dz, dtype="float64")
start, end = (0, 0, 0), (nx, ny, nz)

w = VtkFile("./evtk_test", VtkRectilinearGrid, "ascii", "binary", -1)
w.openGrid(start=start, end=end)
w.openPiece(start=start, end=end)
# Coordinates of cell vertices
w.openElement("Coordinates")
w.addData("x_coordinates", x, False)
w.addData("y_coordinates", y, False)
w.addData("z_coordinates", z, False)
w.closeElement("Coordinates")

# Point data
temp = np.random.rand(npoints)
vx = vy = vz = np.zeros([nx + 1, ny + 1, nz + 1], dtype="float64", order="F")
w.openData("Point", scalars="Temperature", vectors="Velocity")
w.addData("Temperature", temp)
w.addData("Velocity", (vx, vy, vz))
w.closeData("Point")

# Cell data
pressure = np.zeros([nx, ny, nz], dtype="float64", order="F")
w.openData("Cell", scalars="Pressure")
w.addData("Pressure", pressure)
w.closeData("Cell")

w.closePiece() 
w.closeGrid()
w.save()