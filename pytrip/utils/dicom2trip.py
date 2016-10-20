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
from pytrip import CtxCube, dicomhelper


def main(args=sys.argv[1:]):
    if len(args) != 2:
        print("\tusage: dicom2trip folder basename")
        exit()

    basename = args[1]

    # import DICOM
    dcm = dicomhelper.read_dicom_folder(args[0])
    c = CtxCube()
    c.read_dicom(dcm)

    c.write_trip_header(basename + ".hed")
    c.write_trip_data(basename + ".ctx")


if __name__ == '__main__':
    logging.basicConfig()
    sys.exit(main(sys.argv[1:]))
