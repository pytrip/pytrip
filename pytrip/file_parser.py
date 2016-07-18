"""
    This file is part of PyTRiP.

    PyTRiP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyTRiP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyTRiP.  If not, see <http://www.gnu.org/licenses/>
"""


def parse_to_var(data, var, stoptag=""):
    out = {}
    i = 0
    if type(data) is str:
        data = data.split()
    for line in data:
        if line.find(stoptag) > -1:
            break
        items = line.split()
        if items[0] in var:
            if len(items) > 1:
                out[var[items[0]]] = line.replace(items[0] + " ", "")
            else:
                out[var[items[0]]] = True
        i += 1
    return out, i
