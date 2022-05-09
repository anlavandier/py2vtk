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


# =============================================================================
# CELL TYPES
# =============================================================================
class VtkCellType:
    """
    Wrapper class for VTK cell types.
    
    Parameters
    ----------
    tid : int
        Type ID.
    name : str
        Cell type name.
    """

    def __init__(self, tid, name):
        self.tid = tid
        self.name = name

    def __str__(self):
        return "VtkCellType( %s ) \n" % (self.name)

# Linear cells
VtkEmptyCell = VtkCellType(0, "EmptyCell")
VtkVertex = VtkCellType(1, "Vertex")
VtkPolyVertex = VtkCellType(2, "PolyVertex")
VtkLine = VtkCellType(3, "Line")
VtkPolyLine = VtkCellType(4, "PolyLine")
VtkTriangle = VtkCellType(5, "Triangle")
VtkTriangleStrip = VtkCellType(6, "TriangleStrip")
VtkPolygon = VtkCellType(7, "Polygon")
VtkPixel = VtkCellType(8, "Pixel")
VtkQuad = VtkCellType(9, "Quad")
VtkTetra = VtkCellType(10, "Tetra")
VtkVoxel = VtkCellType(11, "Voxel")
VtkHexahedron = VtkCellType(12, "Hexahedron")
VtkWedge = VtkCellType(13, "Wedge")
VtkPyramid = VtkCellType(14, "Pyramid")
VtkPentagonalPrism = VtkCellType(15, "Pentagonal_Prism")
VtkHexagonalPrism = VtkCellType(16, "Hexagonal_Prism")

# Quadratic, isoparametric cells
VtkQuadraticEdge = VtkCellType(21, "Quadratic_Edge")
VtkQuadraticTriangle = VtkCellType(22, "Quadratic_Triangle")
VtkQuadraticQuad = VtkCellType(23, "Quadratic_Quad")
VtkQuadraticTetra = VtkCellType(24, "Quadratic_Tetra")
VtkQuadraticHexahedron = VtkCellType(25, "Quadratic_Hexahedron")
VtkQuadraticWedge = VtkCellType(26, "Quadratic_Wedge")
VtkQuadraticPyramid = VtkCellType(27, "Quadratic_Pyramid")
VtkBiquadraticQuad = VtkCellType(28, "Biquadratic_Quad")
VtkTriquadraticHexahedron = VtkCellType(29, "Triquadratic_Hexahedron")
VtkQuadraticLinearQuad = VtkCellType(30, "Quadratic_Linear_Quad")
VtkQuadraticLinearWedge = VtkCellType(31, "Quadratic_Linear_Wedge")
VtkBiquadraticQuadraticWedge = VtkCellType(32, "Biquadratic_Quadratic_Wedge")
VtkBiquadraticQuadraticHexahedron = VtkCellType(33, "Biquadratic_Quadratic_Hexahedron")
VtkBiquadraticTriangle = VtkCellType(34, "Biquadratic_Triangle")
VtkQuadraticPolygon = VtkCellType(36, "Quadratic_Polygon")
VtkTriquadraticPyramid = VtkCellType(37, "Triquadratic_Pyramid")

# Cubic, isoparametric cell
VtkCubicLine = VtkCellType(35, "Cubic_Line")

# Special class of cells formed by convex group of points
VtkConvexPointSet = VtkCellType(41, "Convex_Point_Set")

# Polyhedron cell (consisting of polygonal faces)
VtkPolyhedron = VtkCellType(42, "Polyhedron")

# Higher order cells in parametric form
VtkParametricCurve = VtkCellType(51, "Parametric_Curve")
VtkParametricSurface = VtkCellType(52, "Parametric_Surface")
VtkParametricTriSurface = VtkCellType(53, "Parametric_Tri_Surface")
VtkParametricQuadSurface = VtkCellType(54, "Parametric_Quad_Surface")
VtkParametricTetraRegion = VtkCellType(55, "¨Parametric_Tetra_Region")
VtkParametricHexRegion = VtkCellType(56, "Parametric_Hex_Region")

# Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
VtkLagrangeCurve = VtkCellType(68, "Lagrange_Curve")
VtkLagrangeTriangle = VtkCellType(69, "Lagrange_Triangle")
VtkLagrangeQuad = VtkCellType(70, "Lagrange_Quad")
VtkLagrangeTetrahedron = VtkCellType(71, "Lagrange_Tetra")
VtkLagrangeHexahedron = VtkCellType(72, "Lagrange_Hexahedron")
VtkLagrangeWedge = VtkCellType(73, "Lagrange_Wedge")
VtkLagrangePyramid = VtkCellType(74, "Lagrange_Pyramid")

# Arbitrary order Bezier elements (formulated separated from generic higher order cells)
VtkBezierCurve = VtkCellType(75, "Bezier_Curve")
VtkBezierTriangle = VtkCellType(76, "Bezier_Triangle")
VtkBezierQuad = VtkCellType(77, "Bezier_Quad")
VtkBezierTetrahedron = VtkCellType(78, "Bezier_Tetra")
VtkBezierHexahedron = VtkCellType(79, "Bezier_Hexahedron")
VtkBezierWedge = VtkCellType(80, "Bezier_Wedge")
VtkLagrangePyramid = VtkCellType(81, "Bezier_Pyramid")

Vtk_points_per_cell = {
    # Linear cells
    0: 0,
    1: 1,
    2: -1,
    3: 2,
    4: -1,
    5: 3,
    6: -1,
    7: -1,
    8: 4,
    9: 4,
    10: 4,
    11: 8,
    12: 8,
    13: 6,
    14: 5,
    15: 10,
    16: 12,

    # Quadratic, isoparametric cells
    21: 3,
    22: 6,
    23: 8,
    24: 10,
    25: 20,
    26: 15,
    27: 13,
    28: 9,
    29: 27,
    30: 6,
    31: 12,
    32: 18,
    33: 24,
    34: 7,
    36: -1,
    37: 20,

    # Cubic, isoparametric cell
    35: -1,
    
    # Special class of cells formed by convex group of points
    41: -1,

    # Polyhedron cell (consisting of polygonal faces)
    42: -1,

    # Higher order cells in parametric form
    51: -1,
    52: -1,
    53: -1,
    54: -1,
    55: -1,
    56: -1,

    # Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
    68: -1,
    69: -1,
    70: -1,
    71: -1,
    72: -1,
    73: -1,
    74: -1,

    # Arbitrary order Bezier elements (formulated separated from generic higher order cells)
    75: -1,
    76: -1,
    77: -1,
    78: -1,
    79: -1,
    80: -1,
    81: -1,    
}