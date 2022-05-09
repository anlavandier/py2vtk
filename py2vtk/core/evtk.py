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


# ***********************************************************************************
# * Copyright © 2018-2022 Lucas Frérot                                              *
# * Permission is hereby granted, free of charge, to any person obtaining a copy of *
# * this software and associated documentation files (the "Software"), to deal in   *
# * the Software without restriction, including without limitation the rights to    * 
# * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies   *
# * of the Software, and to permit persons to whom the Software is furnished to do  *
# * so, subject to the following conditions:                                        *
# * The above copyright notice and this permission notice shall be included in all  *
# * copies or substantial portions of the Software.                                 *
# *                                                                                 *
# * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      *
# * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        * 
# * FITNESS FOR A PARTICULAR  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    *
# * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          *
# * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   *
# * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   *
# * SOFTWARE.                                                                       *                 
# ***********************************************************************************
"""Export routines"""


import io
import struct
import sys
import zlib

import numpy as np

from base64 import b64encode

# Optional packages
try:
    import lzma
except ImportError:
    lzma = None
try:
    import lz4
except ImportError:
    lz4 = None


compressor_dict = {
    'zlib': zlib,
    None: zlib,
    'lzma': lzma,
    'lz4': lz4
}

# Map numpy dtype to struct format
np_to_struct = {
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "float32": "f",
    "float64": "d",
}


def _get_byte_order_char():
    # Check format in https://docs.python.org/3.5/library/struct.html
    if sys.byteorder == "little":
        return "<"
    return ">"

def compress(array, level, compressor):
    """
    Compress an array with a compressor. Taken from uvw, https://github.com/prs513rosewood/uvw
    """
    assert level != 0

    raw_array = memoryview(array.tobytes())
    data_size = array.nbytes

    max_block_size = 2**15
    n_blocks = data_size // max_block_size + 1
    last_block_size = data_size % max_block_size

    # Compress first n_blocks - 1  blocks of size max_block_size
    compressed_data = [zlib.compress(raw_array[i* max_block_size + (i + 1) * max_block_size], level)
                        for i in range(n_blocks - 1)]

    # Compress last block of size last_block_size
    compressed_data.append(zlib.compress(raw_array[-last_block_size:], level=level))

    # Header data (cf https://vtk.org/Wiki/VTK_XML_Formats#Compressed_Data)
    header_dtype = np.dtype(_get_byte_order_char() + 'u8')
    usize = max_block_size
    psize = last_block_size
    csize = [len(x) for x in compressed_data]
    header = np.array([n_blocks, usize, psize] + csize, dtype=header_dtype)

    return header.tobytes(), b"".join(compressed_data)

def encodeData(data, format, level=0):
    """
    Encodes a single numpy ndarray of a 3-tuple of arrays
    using a specific and a compression level if relevant.

    Parameters
    ----------
    data : array or list of arrays of tuple or arrays
        data to encode
    format : str in {}

    Returns
    -------
    size : int
        size in bytes of the encoded data with the header
    encoded_data : bytes
        encoded data with header
    """
    is_vector_comp = isinstance (data, (tuple, list))
    if is_vector_comp:
        assert len(data) == 3

        x, y, z = data[0], data[1], data[2]
        
        assert x.flags['C_CONTIGUOUS'] or x.flags['F_CONTIGUOUS']
        assert y.flags['C_CONTIGUOUS'] or y.flags['F_CONTIGUOUS']
        assert z.flags['C_CONTIGUOUS'] or z.flags['F_CONTIGUOUS']

        assert x.ndim == 1 or x.ndim == 3
        assert y.ndim == 1 or y.ndim == 3
        assert z.ndim == 1 or z.ndim == 3

        assert x.size == y.size == z.size

        x = np.asfortranarray(x)
        y = np.asfortranarray(y)
        z = np.asfortranarray(z)

        xyz = np.empty((3,) + x.shape, dtype=x.dtype, order='F')

        xyz[0] = x
        xyz[1] = y
        xyz[2] = z
    
        xxyyzz = np.ravel(xyz, order='K')

        raw_fmt = (
            _get_byte_order_char() + str(xyz.size) + np_to_struct[x.dtype.name]
        )
        ascii_fmt = '%d' if np.issubdtype(xxyyzz.dtype, np.integer) else '%.18e'

    else:
        x = data

        assert x.flags['C_CONTIGUOUS'] or x.flags['F_CONTIGUOUS']
        
        assert x.ndim == 3 or x.ndim == 1
        
        xx = np.ravel(x, order='F')
        raw_fmt = (
        _get_byte_order_char() + str(x.size) + np_to_struct[x.dtype.name]
        )

        ascii_fmt = '%d' if np.issubdtype(x.dtype, np.integer) else '%.18e'

    if format == "raw":
        size_fmt = (
        _get_byte_order_char() + "Q"
        ) 
        if is_vector_comp:
            bin_pack = struct.pack(size_fmt, xxyyzz.nbytes)
            bin_pack += struct.pack(raw_fmt, *xxyyzz)
            size = xxyyzz.nbytes + 8 # header of size 8
        else:
            bin_pack = struct.pack(size_fmt, xx.nbytes)
            bin_pack +=  struct.pack(raw_fmt, *xx)
            size = xx.nbytes + 8 # header of size 8
        return size, bin_pack
    
    elif format == "ascii":
        stream = io.StringIO()
        if is_vector_comp:
            np.savetxt(
                stream,
                xxyyzz,
                fmt=ascii_fmt,
                newline=' ',
            )
        else:
            np.savetxt(
                stream,
                xx,
                fmt=ascii_fmt,
                newline=' ',
            )
        return  0, stream.getvalue().encode()

    elif format == 'binary':
        if is_vector_comp:
            if level == 0:
                header = np.array(xxyyzz.nbytes, dtype=np.dtype(xxyyzz.dtype.byteorder + 'u8'))
                formatted_data = memoryview(xxyyzz)
            else:
                header, formatted_data = compress(xxyyzz, level)
        else:
            if level == 0:
                header = np.array(xx.nbytes, dtype=np.dtype(xx.dtype.byteorder + 'u8'))
                formatted_data = memoryview(xx)
            else:
                header, formatted_data = compress(xx, level)
 
        encoded_data = b64encode(header) 
        encoded_data += b64encode(formatted_data)
        return len(encoded_data), encoded_data

    else:
        raise ValueError(f"format {format} is not understood")
