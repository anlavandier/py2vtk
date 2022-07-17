import pytest

import numpy as np

from vtk import (
    vtkXMLRectilinearGridReader,
    vtkXMLImageDataReader,
    vtkXMLStructuredGridReader,
    vtkXMLUnstructuredGridReader,
)

from py2vtk.core.vtkfiles import (VtkFile,
                                  VtkParallelFile,
                                  VtkImageData,
                                  VtkPImageData,
                                  VtkRectilinearGrid,
                                  VtkPRectilinearGrid,
                                  VtkPolyData,
                                  VtkPPolyData,
                                  VtkStructuredGrid,
                                  VtkPStructuredGrid,
                                  VtkUnstructuredGrid,
                                  VtkPUnstructuredGrid)


@pytest.mark.parametrize('compressor', ['zlib', 'lzma'])
@pytest.mark.parametrize('compression', ['True', 'False', 5])
@pytest.mark.parametrize('direct_format', ['binary', 'ascii'])
@pytest.mark.parametrize('appended_format', ['binary', 'raw'])
def test_image_data(compressor, compression, direct_format, appended_format):

