#
#    Copyright (C) 2010-2017 PyTRiP98 Developers.
#
#    This file is part of PyTRiP98.
#
#    PyTRiP98 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyTRiP98 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyTRiP98.  If not, see <http://www.gnu.org/licenses/>.
#
"""
This module contains a parser method for parsing voxelplan files.
"""


def parse_to_var(data, var, stoptag=""):
    """ Parses a variable 'var' from 'data'.

    :returns: (out,i) tuple
    out - the rest of the line with 'var' and a " " removed. If nothing is followed, then True is returned.
    i - number of lines parsed.
    """
    out = {}
    i = 0
    if type(data) is str:
        data = data.split()
    for line in data:
        if line.find(stoptag) > -1:
            break
        items = line.split()
        if items[0] in var:  # if we have found 'var' in this line..
            if len(items) > 1:
                # remove "var" from that line, and returns the rest of that line.
                out[var[items[0]]] = line.replace(items[0] + " ", "")
            else:
                # in case nothing followed var. The presences of 'var' means smth is set to True.
                out[var[items[0]]] = True
        i += 1
    return out, i
