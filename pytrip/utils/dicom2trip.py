#!/usr/bin/env python
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
Script for converting a DICOM file to TRiP98 / Voxelplan style files.
"""
import sys
import logging
import argparse

import pytrip as pt


def main(args=sys.argv[1:]):
    """ Main function for dicom2trip.py
    """
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dicom_folder", help="location of folder with DICOM files", type=str)
    parser.add_argument("ctx_basename", help="basename of output file in TRiP98 format", type=str)
    parser.add_argument("-v", "--verbosity", action='count', help="increase output verbosity", default=0)
    parser.add_argument('-V', '--version', action='version', version=pt.__version__)
    args = parser.parse_args(args)

    basename = args.ctx_basename

    # import DICOM
    dcm = pt.dicomhelper.read_dicom_folder(args.dicom_folder)
    c = pt.CtxCube()
    c.read_dicom(dcm)

    c.write_trip_header(basename + ".hed")
    c.write_trip_data(basename + ".ctx")
    return 0


if __name__ == '__main__':
    logging.basicConfig()
    sys.exit(main(sys.argv[1:]))
