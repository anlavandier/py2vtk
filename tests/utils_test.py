from vtkmodules.util.numpy_support import vtk_to_numpy


def get_vtk_data(reader, filepath):
    reader.SetFileName(filepath)
    reader.Update()
    output = reader.GetOutput()
    return (
        vtk_to_numpy(output.GetPointData().GetArray("points")),
        vtk_to_numpy(output.GetCellData().GetArray("cells")),
        vtk_to_numpy(output.GetFieldData().GetArray("field")),
    )
