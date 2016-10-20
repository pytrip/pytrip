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
import argparse

import pytrip as pt
from pytrip.utils.rst_read import RstfileRead


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("rst_file", help="location of rst file in TRiP98 format", type=str)
    parser.add_argument("sobp_file", help="location of SHIELD-HIT12A sobp.file to write", type=str)
    parser.add_argument("-v", "--verbosity", action='count', help="increase output verbosity", default=0)
    parser.add_argument('-V', '--version', action='version', version=pt.__version__)
    args = parser.parse_args(args)

    rst_file_data = RstfileRead(args.rst_file)
    with open(args.sobp_file, 'w') as fout:
        fout.writelines("*ENERGY(GEV) X(CM)  Y(CM)     FWHM(cm)  WEIGHT\n")
        for subm in rst_file_data.submachine:
            for xpos, ypos, part in zip(subm.xpos, subm.ypos, subm.particles):
                fout.writelines("{:<10.6f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.4e}\n".format(
                    subm.energy / 1000.0, xpos / 10.0, ypos / 10.0, subm.focus / 10.0, part))
    return 0


if __name__ == '__main__':
    logging.basicConfig()
    sys.exit(main(sys.argv[1:]))
