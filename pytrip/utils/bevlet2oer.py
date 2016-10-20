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
import argparse
import logging

from scipy import interpolate

import pytrip as pt


class ReadGd(object):
    '''read file'''

    def __init__(self, gd_filename, _dataset=0, dat_filename=None):

        if os.path.isfile(gd_filename) is False:
            raise IOError("Could not find file " + gd_filename)

        if _dataset > 2:
            print("DOS: Error- only 0,1,2 OER set available. Got:", _dataset)
        path = os.path.dirname(__file__)
        path_data = (os.path.join(path, "data/OER_barendsen.dat"), os.path.join(path, "data/OER_furusawa_HSG_C12.dat"),
                     os.path.join(path, "data/OER_furusawa_V79_C12.dat"))
        fd = open(path_data[_dataset], 'r')
        lines = fd.readlines()
        fd.close()
        x = [line.split()[0] for line in lines]
        y = [line.split()[1] for line in lines]
        us = interpolate.UnivariateSpline(x, y, s=0.0)

        gd_file = open(gd_filename, 'r')
        gd_lines = gd_file.readlines()
        gd_file.close()
        first = True
        ignore_rest = False

        if dat_filename is not None:
            out_fd = open(dat_filename, 'w')
        else:
            out_fd = sys.stdout

        for line in gd_lines:
            if not (line[0].isdigit()):
                tmp_string = "#" + line
                if not first:
                    ignore_rest = True
            else:
                first = False
                if ignore_rest:
                    tmp_string = "#" + line
                else:
                    let = line.split()[7]
                    oer = us(let)
                    tmp_string = ""
                    for item in line.split():
                        tmp_string = tmp_string + item + " "
                    tmp_string = tmp_string + str(oer[0]) + "\n"

            out_fd.write(tmp_string)

        if dat_filename is not None:
            out_fd.close()


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("gd_file", help="location of bevlet gd file", type=str)
    parser.add_argument("dat_file", help="location of OER .dat to write", type=str, nargs='?')
    parser.add_argument("-v", "--verbosity", action='count', help="increase output verbosity", default=0)
    parser.add_argument('-V', '--version', action='version', version=pt.__version__)
    args = parser.parse_args(args)

    ReadGd(args.gd_file, 2, args.dat_file)

    return 0

if __name__ == '__main__':
    logging.basicConfig()
    sys.exit(main(sys.argv[1:]))
