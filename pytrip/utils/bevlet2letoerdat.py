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
Convert bevlet (Beams Eye View LET) to OER (Oxygen Enhancement Ratio) values.
"""
import sys
import os
import logging


class ReadGd(object):
    '''read file'''

    def __init__(self, filename):

        if os.path.isfile(filename) is False:
            raise IOError("Could not find file " + filename)

        gd_file = open(filename, 'r')
        gd_lines = gd_file.readlines()
        gd_file.close()
        first = True
        ignore_rest = False
        for line in gd_lines:
            if not (line[0].isdigit()):
                string = "#" + line
                if not first:
                    ignore_rest = True
            else:
                first = False
                if ignore_rest:
                    string = "#" + line
                else:
                    string = line

            sys.stdout.write(string)


if __name__ == '__main__':
    logging.basicConfig()
    ReadGd(sys.argv[1])
