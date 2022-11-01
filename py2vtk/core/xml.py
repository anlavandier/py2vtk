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
"""Simple class to generate a well-formed XML file"""

# Compared to pyevtk's original XmlWriter class, automatic
# tabulation (using 2 whitespaces) was added to make the XML file more
# human readable.


class XmlWriter:
    """
    Xml writer class.

    Parameters
    ----------

    filepath : str
        Path to the xml file.

    addDeclaration : bool, optional
        Whether to add the declaration.
        The default is True.

    """

    def __init__(self, filepath, addDeclaration=True):
        self.stream = open(filepath, "wb")
        self.openTag = False
        self.current = []
        if addDeclaration:
            self.addDeclaration()

    def close(self):
        """
        Close the file.

        """
        assert not self.openTag
        self.stream.close()

    def addDeclaration(self):
        """
        Add xml declaration.

        """
        self.stream.write(b'<?xml version="1.0"?>')

    def openElement(self, tag):
        """
        Open a new xml element.

        """
        if self.openTag:
            self.stream.write(b">")
        st = f"\n{' '*len(self.current) * 2}<{tag}"
        self.stream.write(str.encode(st))
        self.openTag = True
        self.current.append(tag)
        return self

    def closeElement(self, tag=None):
        """
        Close the current element.

        Parameters
        ----------

        tag : str, optional
            Tag of the element.
            The default is None.

        Returns
        -------

        XmlWriter
            The XmlWriter itself for chained calls.

        """
        if tag:
            assert self.current.pop() == tag
            if self.openTag:
                self.stream.write(b">")
                self.openTag = False
            st = f"\n{' '*len(self.current) * 2}</{tag}>"
            self.stream.write(str.encode(st))
        else:
            self.stream.write(b"/>")
            self.openTag = False
            self.current.pop()
        return self

    def addText(self, text):
        """
        Add text.

        Parameters
        ----------

        text : str
            Text to add.

        Returns
        -------

        XmlWriter
            The XmlWriter itself for chained calls.

        """
        if self.openTag:
            self.stream.write(b">\n")
            self.openTag = False
        self.stream.write(str.encode(text))
        return self

    def addAttributes(self, newline=False, **kwargs):
        """
        Add attributes.

        Parameters
        ----------

        newline : bool, default=False
            If True, adds a ">" and newline after adding all of the attributes.

        kwargs : dict
            keys as attribute names.

        Returns
        -------

        XmlWriter
            The XmlWriter itself for chained calls.

        """
        assert self.openTag
        for key in kwargs:
            st = f' {key}="{kwargs[key]}"'
            self.stream.write(str.encode(st))
        if newline:
            newline = ">\n" + " " * len(self.current) * 2
            self.stream.write(str.encode(newline))
            self.openTag = False
        return self
