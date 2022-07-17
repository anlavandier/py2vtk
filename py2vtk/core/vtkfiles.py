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
"""Definition of the VTK File formats and low level interface"""

import sys
import os
import numpy as np

from .evtk import encodeData
from .xml import XmlWriter

__all__ = ['VtkImageData',
           'VtkPolyData',
           'VtkRectilinearGrid',
           'VtkStructuredGrid',
           'VtkUnstructuredGrid',
           'VtkPImageData',
           'VtkPPolyData',
           'VtkPRectilinear',
           'VtkPStructuredGrid',
           'VtkPUnstructuredGrid',
           'VtkFile',
           'VtkParallelFile']

# =============================================================================
# VTK Filetypes
# =============================================================================
# Serial
class VtkFileType:
    """
    Wrapper class for serial VTK file types.
    
    Parameters
    ----------
    name : str
        Data name.
    ext : str
        File extension.
    """

    def __init__(self, name, ext):
        self.name = name
        self.ext = ext

    def __str__(self):
        return "Name: %s  Ext: %s \n" % (self.name, self.ext)


VtkImageData = VtkFileType("ImageData", ".vti")
VtkPolyData = VtkFileType("PolyData", ".vtp")
VtkRectilinearGrid = VtkFileType("RectilinearGrid", ".vtr")
VtkStructuredGrid = VtkFileType("StructuredGrid", ".vts")
VtkUnstructuredGrid = VtkFileType("UnstructuredGrid", ".vtu")


# Parallel
class VtkParallelFileType:
    """
    A wrapper class for parallel VTK file types.
    
    Parameters
    ----------
    vtkftype : VtkFileType
        Vtk file type
    """

    def __init__(self, vtkftype):
        self.name = "P" + vtkftype.name
        ext = vtkftype.ext
        self.ext = ext[0] + "p" + ext[1:]

    def __str__(self):
        return "Name: %s  Ext: %s \n" % (self.name, self.ext)


VtkPImageData = VtkParallelFileType(VtkImageData)
VtkPPolyData = VtkParallelFileType(VtkPolyData)
VtkPRectilinearGrid = VtkParallelFileType(VtkRectilinearGrid)
VtkPStructuredGrid = VtkParallelFileType(VtkStructuredGrid)
VtkPUnstructuredGrid = VtkParallelFileType(VtkUnstructuredGrid)

# =============================================================================
# VTK Data Types
# =============================================================================
class VtkDataType:
    """
    Wrapper class for VTK data types.
    
    Parameters
    ----------
    size : int
        Size in byte.
    name : str
        Type name.
    """

    def __init__(self, size, name):
        self.size = size
        self.name = name

    def __str__(self):
        return "Type: %s  Size: %d \n" % (self.name, self.size)


VtkInt8 = VtkDataType(1, "Int8")
VtkUInt8 = VtkDataType(1, "UInt8")
VtkInt16 = VtkDataType(2, "Int16")
VtkUInt16 = VtkDataType(2, "UInt16")
VtkInt32 = VtkDataType(4, "Int32")
VtkUInt32 = VtkDataType(4, "UInt32")
VtkInt64 = VtkDataType(8, "Int64")
VtkUInt64 = VtkDataType(8, "UInt64")
VtkFloat32 = VtkDataType(4, "Float32")
VtkFloat64 = VtkDataType(8, "Float64")


# Map numpy to VTK data types
np_to_vtk = {
    "int8": VtkInt8,
    "uint8": VtkUInt8,
    "int16": VtkInt16,
    "uint16": VtkUInt16,
    "int32": VtkInt32,
    "uint32": VtkUInt32,
    "int64": VtkInt64,
    "uint64": VtkUInt64,
    "float32": VtkFloat32,
    "float64": VtkFloat64,
}



# ==============================
# Helper functions
# ==============================
def _mix_extents(start, end):
    assert len(start) == len(end) == 3
    string = "%d %d %d %d %d %d" % (
        start[0],
        end[0],
        start[1],
        end[1],
        start[2],
        end[2],
    )
    return string


def _array_to_string(a):
    s = "".join([repr(num) + " " for num in a])
    return s


def _get_byte_order():
    if sys.byteorder == "little":
        return "LittleEndian"
    return "BigEndian"


# ================================
#        VtkGroup class
# ================================
# class VtkGroup:
#     """
#     Creates a VtkGroup file that is stored in filepath.
#     Parameters
#     ----------
#     filepath : str
#         filename without extension.
#     """

#     def __init__(self, filepath):
#         self.xml = XmlWriter(filepath + ".pvd")
#         self.xml.openElement("VTKFile")
#         self.xml.addAttributes(
#             type="Collection", version="0.1", byte_order=_get_byte_order()
#         )
#         self.xml.openElement("Collection")
#         self.root = os.path.dirname(filepath)

#     def save(self):
#         """Close this VtkGroup."""
#         self.xml.closeElement("Collection")
#         self.xml.closeElement("VTKFile")
#         self.xml.close()

#     def addFile(self, filepath, sim_time, group="", part="0", name=""):
#         """
#         Add a file to this VTK group.
#         Parameters
#         ----------
#         filepath : str
#             full path to VTK file.
#         sim_time : float
#             simulated time.
#         group : str, optional
#             This attribute is not required;
#             it is only for informational purposes.
#             The default is "".
#         part : int, optional
#             It is an integer value greater than or equal to 0.
#             The default is "0".
#         Notes
#         -----
#         See: http://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format for details.
#         """
#         # TODO: Check what the other attributes are for.
#         filename = os.path.relpath(filepath, start=self.root)
#         self.xml.openElement("DataSet")
#         self.xml.addAttributes(
#             timestep=sim_time, group=group, part=part, file=filename, name=name
#         )
#         self.xml.closeElement()


# ================================
# VtkFile class
# ================================
# Serial
class VtkFile:
    """
    Class for a serial VTK file.

    Parameters
    ----------
    filepath : str
        filename without extension.
    ftype : str
        file type, e.g. VtkImageData, etc.
    
    direct_format: str in {"ascii", "binary"}, default="binary"
        format of the non appended data.

    appended_format : str in {"base64", "raw"}, default="raw"
        format of the appended data.
    
    compression : bool or int, default=False
        Whether and how much to compress the binary data.
    
    compressor : str in {'zlib', 'lzma'}, default='zlib'
        compressor to use to compress binary data.
    """
    def __init__(self, filepath, ftype, direct_format="ascii", appended_format="raw", compression=False,
                 compressor='zlib'):
        self.ftype = ftype
        self.filename = filepath + ftype.ext
        self.xml = XmlWriter(self.filename)
        self.offset = 0  # offset in bytes after beginning of binary section

        # Formatting
        assert direct_format in ["ascii", "binary"]
        self.direct_format = direct_format
        assert appended_format in ["binary", "raw"]
        self.appended_format = appended_format
        self.data_to_append = []

        self.appendedDataIsOpen = False
        self.isOpen = True

        self.xml.openElement("VTKFile").addAttributes(
            type=ftype.name,
            version="1.0",
            byte_order=_get_byte_order(),
            header_type="UInt64",
        )

        if not compression:
            self.compression = 0
        elif compression is True:
            self.compression = -1
        elif isinstance(compression, int) and -1 <= compression and compression <= 9:
            self.compression = compression
        else:
            raise ValueError("compression can only be True, False, or an"
                             "integer between -1 and 9 included")
        
        if self.compression != 0:
            if compressor == 'zlib':
                self.xml.addAttributes(compressor='vtkZLibDataCompressor')
            elif compressor == 'lzma':
                self.xml.addAttributes(compressor='vtkLZMADataCompressor')
            else:
                raise ValueError("Invalid compressor name")

        self.compressor = compressor


    def getFileName(self):
        """Return absolute path to this file."""
        return os.path.abspath(self.filename)

    def openPiece(
        self,
        start=None,
        end=None,
        npoints=None,
        ncells=None,
        nverts=None,
        nlines=None,
        nstrips=None,
        npolys=None,
    ):
        """
        Open piece section.
        
        Parameters
        ----------
        start : array-like, optional
            array or list with start indexes in each direction.
            Must be given with end.
        end : array-like, optional
            array or list with end indexes in each direction.
            Must be given with start.
        npoints : int, optional
            number of points in piece
        ncells : int, optional
            number of cells in piece.
            If present, npoints must also be given.
        nverts : int, optional
            number of vertices.
        nlines : int, optional
            number of lines.
        nstrips : int, optional
            number of stripes.
        npolys : int, optional
            number of poly.
        """
        self.xml.openElement("Piece")
        if start and end:
            ext = _mix_extents(start, end)
            self.xml.addAttributes(Extent=ext)

        elif start and not end:
            raise ValueError("start and end must be provided together")
        elif not start and end:
            raise ValueError("start and end must be provided together")

        elif ncells and npoints:
            self.xml.addAttributes(NumberOfPoints=npoints, NumberOfCells=ncells)

        elif npoints or nverts or nlines or nstrips or npolys:
            if npoints is None:
                npoints = str(0)
            if nverts is None:
                nverts = str(0)
            if nlines is None:
                nlines = str(0)
            if nstrips is None:
                nstrips = str(0)
            if npolys is None:
                npolys = str(0)
            self.xml.addAttributes(
                NumberOfPoints=npoints,
                NumberOfVerts=nverts,
                NumberOfLines=nlines,
                NumberOfStrips=nstrips,
                NumberOfPolys=npolys,
            )
        else:
            raise ValueError("Incorrect argument")

    def closePiece(self):
        """Close Piece."""
        self.xml.closeElement("Piece")

    def openData(
        self,
        nodeType,
        scalars=None,
        vectors=None,
        normals=None,
        tensors=None,
        tcoords=None,
    ):
        """
        Open data section.
        
        Parameters
        ----------
        nodeType : str
            Either "Point", "Cell" or "Field".
        scalars : str, optional
            default data array name for scalar data.
        vectors : str, optional
            default data array name for vector data.
        normals : str, optional
            default data array name for normals data.
        tensors : str, optional
            default data array name for tensors data.
        tcoords : str, optional
            default data array name for tcoords data.
        """
        self.xml.openElement(nodeType + "Data")
        if scalars:
            self.xml.addAttributes(Scalars=scalars)
        if vectors:
            self.xml.addAttributes(Vectors=vectors)
        if normals:
            self.xml.addAttributes(Normals=normals)
        if tensors:
            self.xml.addAttributes(Tensors=tensors)
        if tcoords:
            self.xml.addAttributes(TCoords=tcoords)

    def closeData(self, nodeType):
        """
        Close data section.
        
        Parameters
        ----------
        nodeType : str
            "Point", "Cell" or "Field".
        """
        self.xml.closeElement(nodeType + "Data")

    def openGrid(self, start=None, end=None, origin=None, spacing=None):
        """
        Open grid section.
        
        Parameters
        ----------
        start : array-like, optional
            array or list of start indexes.
            Required for Structured, Rectilinear and ImageData grids.
            The default is None.
        end : array-like, optional
            array or list of end indexes.
            Required for Structured, Rectilinear and ImageData grids.
            The default is None.
        origin : array-like, optional
            3D array or list with grid origin.
            Only required for ImageData grids.
            The default is None.
        spacing : array-like, optional
            3D array or list with grid spacing.
            Only required for ImageData grids.
            The default is None.
        """
        gType = self.ftype.name
        self.xml.openElement(gType)
        if gType == VtkImageData.name:
            if not start or not end or not origin or not spacing:
                raise ValueError(f"{gType} requires start, end, origin and spacing")
            ext = _mix_extents(start, end)
            self.xml.addAttributes(
                WholeExtent=ext,
                Origin=_array_to_string(origin),
                Spacing=_array_to_string(spacing),
            )

        elif gType in [VtkStructuredGrid.name, VtkRectilinearGrid.name]:
            if not start or not end:
                raise ValueError(f"{gType} requires start and end")
            ext = _mix_extents(start, end)
            self.xml.addAttributes(WholeExtent=ext)

    def closeGrid(self):
        """
        Close grid element.
        """
        self.xml.closeElement(self.ftype.name)



    def addData(self, name, data, append=True):
        """
        Add array description to xml header section and
        writes the array directly if the append is False

        Parameters
        ----------
        name : str
            data array name.
        data : array-like
            one numpy array or a tuple with 3 numpy arrays.
            If a tuple, the individual arrays must represent the components
            of a vector field.
            All arrays must be one dimensional or three-dimensional.
        format : str in {"appended", "ascii", "binary"}
        """        
        if isinstance(data, (tuple, list)):
            assert len(data) == 3
            ncomp = 3
            nelem = data[0].size
            dtype = data[0].dtype.name
        
        else:
            assert isinstance(data, np.ndarray)
            ncomp = 1
            nelem = data.size
            dtype = data.dtype.name

        dtype = np_to_vtk[dtype]

        self.xml.openElement("DataArray")
        if not append:
            self.xml.addAttributes(
                newline=True,
                Name=name,
                type=dtype.name,
                NumberOfComponents=ncomp,
                NumberOfTuples=nelem,
                format=self.direct_format,
            )
            _, encoded_data = encodeData(data, self.direct_format, 
                                         level=self.compression, 
                                         compressor=self.compressor)

            self.xml.stream.write(encoded_data)
            self.closeElement("DataArray")
        else:
            size, encoded_data = encodeData(data, self.appended_format, 
                                            level=self.compression, 
                                            compressor=self.compressor)
            self.xml.addAttributes(
                Name=name,
                type=dtype.name,
                NumberOfComponents=ncomp,
                NumberOfTuples=nelem,
                format="appended",
                offset=self.offset
            )
            self.xml.closeElement()
            self.offset += size
            self.data_to_append.append(encoded_data)

    def _appendData(self):
        """
        Append data to binary section.
        This function writes the header section
        and the data to the binary file.
        Parameters
        ----------
        data : array-like
            one numpy array or a tuple with 3 numpy arrays.
            If a tuple, the individual
            arrays must represent the components of a vector field.
            All arrays must be one dimensional or three-dimensional.
            The order of the arrays must coincide with
            the numbering scheme of the grid.
        """
        self._openAppendedData()
        for i in range(len(self.data_to_append)):
            data_to_append = self.data_to_append.pop(0)
            self.xml.stream.write(data_to_append)

    def _openAppendedData(self):
        """
        Open binary section.
        It is not necessary to explicitly call this function
        from an external library.
        """
        if not self.appendedDataIsOpen:
            if self.appended_format != "raw":
                self.xml.openElement("AppendedData").addAttributes(format="base64").addText(
                    "_"
                )            
            else:
                self.xml.openElement("AppendedData").addAttributes(encoding="raw").addText(
                    "_"
                )
            self.appendedDataIsOpen = True

    def _closeAppendedData(self):
        """
        Close binary section.
        It is not necessary to explicitly call this function
        from an external library.
        """
        self.xml.closeElement("AppendedData")

    def openElement(self, tagName):
        """
        Open an element.
        Useful to add elements such as: Coordinates, Points, Verts, etc.
        """
        self.xml.openElement(tagName)

    def closeElement(self, tagName):
        """Close an element."""
        self.xml.closeElement(tagName)

    def save(self):
        """Close file."""
        if self.data_to_append != []:
            self._appendData()
            self._closeAppendedData()
        self.xml.closeElement("VTKFile")
        self.xml.close()
        self.isOpen = False

    def __del__(self):
        if self.isOpen:
            self.save()


# ================================
#        VtkParallelFile class
# ================================
class VtkParallelFile:
    """
    Class for a VTK parallel file.

    Parameters
    ----------
    filepath : str
        filename without extension

    ftype : VtkParallelFileType
    """

    def __init__(self, filepath, ftype):
        assert isinstance(ftype, VtkParallelFileType)
        self.ftype = ftype
        self.filename = filepath + ftype.ext
        self.xml = XmlWriter(self.filename)
        self.xml.openElement("VTKFile").addAttributes(
            type=ftype.name,
            version="1.0",
            byte_order=_get_byte_order(),
            header_type="UInt64",
        )
        self.isOpen = True


    def getFileName(self):
        """Return absolute path to this file."""
        return os.path.abspath(self.filename)

    def addPiece(
        self,
        source=None,
        start=None,
        end=None,
    ):
        """
        Add piece section with extent and source.

        Parameters
        ----------
        start : array-like, optional
            array or list with start indexes in each direction.
            Must be given with end.
        end : array-like, optional
            array or list with end indexes in each direction.
            Must be given with start.
        source : str
            Source of this piece
        Returns
        -------
        VtkParallelFile
            This VtkFile to allow chained calls.
        """
        # Check Source
        assert source is not None
        assert source.split(".")[-1] == self.ftype.ext[2:]

        self.xml.openElement("Piece")
        if start and end:
            ext = _mix_extents(start, end)
            self.xml.addAttributes(Extent=ext)
        self.xml.addAttributes(Source=source)
        self.xml.closeElement()
        return self

    def openData(
        self,
        nodeType,
        scalars=None,
        vectors=None,
        normals=None,
        tensors=None,
        tcoords=None,
    ):
        """
        Open data section.

        Parameters
        ----------
        nodeType : str
            Either "Point", "Cell" or "Field".
        scalars : str, optional
            default data array name for scalar data.
        vectors : str, optional
            default data array name for vector data.
        normals : str, optional
            default data array name for normals data.
        tensors : str, optional
            default data array name for tensors data.
        tcoords : str, optional
            default data array name for tcoords data.

        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.openElement(nodeType + "Data")
        if scalars:
            self.xml.addAttributes(Scalars=scalars)
        if vectors:
            self.xml.addAttributes(Vectors=vectors)
        if normals:
            self.xml.addAttributes(Normals=normals)
        if tensors:
            self.xml.addAttributes(Tensors=tensors)
        if tcoords:
            self.xml.addAttributes(TCoords=tcoords)

        return self

    def closeData(self, nodeType):
        """
        Close data section.

        Parameters
        ----------
        nodeType : str
            "Point", "Cell" or "Field".

        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.closeElement(nodeType + "Data")

    def openGrid(self, start=None, end=None, origin=None, spacing=None, ghostlevel=0):
        """
        Open grid section.
        Parameters
        ----------
        start : array-like, optional
            array or list of start indexes.
            Required for Structured, Rectilinear and ImageData grids.
            The default is None.
        end : array-like, optional
            array or list of end indexes.
            Required for Structured, Rectilinear and ImageData grids.
            The default is None.
        origin : array-like, optional
            3D array or list with grid origin.
            Only required for ImageData grids.
            The default is None.
        spacing : array-like, optional
            3D array or list with grid spacing.
            Only required for ImageData grids.
            The default is None.
        ghostlevel : int
            Number of ghost-levels by which
            the extents in the individual pieces overlap.
        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        gType = self.ftype.name
        self.xml.openElement(gType)

        if gType == VtkPImageData.name:
            if not start or not end or not origin or not spacing:
                raise ValueError(f"start, end, origin and spacing required for {gType}")
            ext = _mix_extents(start, end)
            self.xml.addAttributes(
                WholeExtent=ext,
                Origin=_array_to_string(origin),
                Spacing=_array_to_string(spacing),
            )

        elif gType in [VtkPStructuredGrid.name, VtkPRectilinearGrid.name]:
            if not start or not end:
                raise ValueError(f"start and end required for {gType}.")
            ext = _mix_extents(start, end)
            self.xml.addAttributes(WholeExtent=ext)

        # Ghostlevel
        self.xml.addAttributes(Ghostlevel=ghostlevel)
        return self

    def closeGrid(self):
        """
        Close grid element.
        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.closeElement(self.ftype.name)

    def addData(self, name, dtype, ncomp):
        """
        Add data array to the parallel vtk file.

        Parameters
        ----------
        name : str
            data array name.
        dtype : str
            data type.
        ncomp : int
            number of components, 1 (=scalar) and 3 (=vector).
        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.

        """
        dtype = np_to_vtk[dtype.name]

        self.xml.openElement("DataArray")
        self.xml.addAttributes(
            Name=name,
            NumberOfComponents=ncomp,
            type=dtype.name,
        )
        self.xml.closeElement()

    def openElement(self, tagName):
        """
        Open an element.
        Useful to add elements such as: Coordinates, Points, Verts, etc.
        """
        self.xml.openElement(tagName)

    def closeElement(self, tagName):
        self.xml.closeElement(tagName)

    def save(self):
        """Close file."""
        self.xml.closeElement("VTKFile")
        self.xml.close()
        self.isOpen = False

    def __del__(self):
        if self.isOpen:
            self.save()