from vtkmodules.util.numpy_support import vtk_to_numpy


def get_vtk_data(reader, filepath, point="points", cell="cells", field="field"):
    reader.SetFileName(filepath)
    reader.Update()
    output = reader.GetOutput()
    pointdata = {}
    celldata = {}
    fielddata = {}

    if isinstance(point, str):
        pointdata = vtk_to_numpy(output.GetPointData().GetArray(point))
    elif isinstance(point, (tuple, list)):
        for p in point:
            pointdata[p] = vtk_to_numpy(output.GetPoinData().GetArray(p))

    if isinstance(cell, str):
        celldata = vtk_to_numpy(output.GetCellData().GetArray(cell))
    elif isinstance(cell, (tuple, list)):
        for c in cell:
            celldata[c] = vtk_to_numpy(output.GetCellData().GetArray(c))

    if isinstance(field, str):
        fielddata = vtk_to_numpy(output.GetFieldData().GetArray(field))
    elif isinstance(field, (tuple, list)):
        for f in field:
            fielddata[f] = vtk_to_numpy(output.GetFieldData().GetArray(f))

    return (pointdata, celldata, fielddata)
