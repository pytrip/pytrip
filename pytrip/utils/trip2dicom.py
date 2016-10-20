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
Script for converting Voxelplan / TRiP98 Ctx and Vdx data to a DICOM file.
"""
import os
import sys
import logging

from pytrip import CtxCube, VdxCube


def main(args=sys.argv[1:]):
    print("\ttrip2dicom is a part of pytrip which was developed by \n\t"
          "Niels Bassler (bassler@phys.au.dk) and \n\t"
          "Jakob Toftegaard (jakob.toftegaard@gmail.com)")

    if len(args) != 2:
        print("\tusage: trip2dicom.py headerfile output_folder")
        print("\ttrip2dicom.py tripfile.hed dicomfolder/")
        exit()

    header_file_input_name = args[0]
    header_basename = os.path.splitext(header_file_input_name)[0]
    output_folder = args[1]

    _, data_file_name = CtxCube.parse_path(header_file_input_name)
    data_file_path = CtxCube.discover_file(data_file_name)

    if not os.path.exists(data_file_path):
        print("CTX file missing")
        exit()

    print("Convert CT images")
    c = CtxCube()
    c.read(header_basename)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    c.write_dicom(output_folder)

    if os.path.exists(header_basename + ".vdx"):
        print("Convert structures")
        v = VdxCube(c)
        v.read(header_basename + ".vdx")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        v.write_dicom(output_folder)
    print("Done")


if __name__ == '__main__':
    logging.basicConfig()
    sys.exit(main(sys.argv[1:]))
