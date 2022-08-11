# ***********************************************************************************
# * Copyright 2010 - 2016 Paulo A. Herrera. All rights reserved.                    *
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
import numpy as np

from ..core.vtkfiles import VtkFile, VtkParallelFile

__all__ = ["_addDataToFile", "_addFieldDataToFile", "_addDataToParallelFile"]


def _addDataToFile(vtkFile, cellData=None, pointData=None, append=True):
    assert isinstance(vtkFile, VtkFile)
    # Point data
    if pointData:
        keys = list(pointData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next(
            (key for key in keys if isinstance(pointData[key], np.ndarray)), None
        )
        vectors = next((key for key in keys if isinstance(pointData[key], tuple)), None)
        vtkFile.openData("Point", scalars=scalars, vectors=vectors)
        for key in keys:
            data = pointData[key]
            vtkFile.addData(key, data, append=append)
        vtkFile.closeData("Point")

    # Cell data
    if cellData:
        keys = list(cellData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next(
            (key for key in keys if isinstance(cellData[key], np.ndarray)), None
        )
        vectors = next((key for key in keys if isinstance(cellData[key], tuple)), None)
        vtkFile.openData("Cell", scalars=scalars, vectors=vectors)
        for key in keys:
            data = cellData[key]
            vtkFile.addData(key, data, append=append)
        vtkFile.closeData("Cell")


def _addFieldDataToFile(vtkFile, fieldData=None, append=True):
    # Field data
    # https://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files#XML_VTK_files
    if fieldData:
        vtkFile.openData("Field")  # no attributes in FieldData
        for key, data in fieldData.items():
            vtkFile.addData(key, data, append=append)
        vtkFile.closeData("Field")


def _addDataToParallelFile(vtkParallelFile, cellData, pointData):
    assert isinstance(vtkParallelFile, VtkParallelFile)
    # Point data
    if pointData:
        keys = list(pointData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next((key for key in keys if pointData[key][1] == 1), None)
        vectors = next((key for key in keys if pointData[key][1] == 3), None)
        vtkParallelFile.openData("PPoint", scalars=scalars, vectors=vectors)
        for key in keys:
            dtype, ncomp = pointData[key]
            vtkParallelFile.addHeader(key, dtype=dtype, ncomp=ncomp)
        vtkParallelFile.closeData("PPoint")

    # Cell data
    if cellData:
        keys = list(cellData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next((key for key in keys if cellData[key][1] == 1), None)
        vectors = next((key for key in keys if cellData[key][1] == 3), None)
        vtkParallelFile.openData("PCell", scalars=scalars, vectors=vectors)
        for key in keys:
            dtype, ncomp = cellData[key]
            vtkParallelFile.addHeader(key, dtype=dtype, ncomp=ncomp)
        vtkParallelFile.closeData("PCell")
