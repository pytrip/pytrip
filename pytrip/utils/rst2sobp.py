#! /usr/bin/env python
#
#    Copyright (C) 2010-2016 PyTRiP98 Developers.
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
TODO: documentation here.
"""
import sys
import logging
from pytrip.utils.rst_read import RstfileRead


def main(args=sys.argv[1:]):
    file = args[0]
    a = RstfileRead(file)
    fout = open("sobp.dat", 'w')
    for i in range(a.submachines):
        b = a.submachine[i]
        for j in range(len(b.xpos)):
            fout.writelines("%-10.6f%-10.2f%-10.2f%-10.2f%-10.4e\n" % (b.energy / 1000.0, b.xpos[j] / 10.0, b.ypos[j] /
                                                                       10.0, b.focus / 10.0, b.particles[j]))
    fout.close()


if __name__ == '__main__':
    logging.basicConfig()
    sys.exit(main(sys.argv[1:]))
