#!/usr/bin/env python

import os
import sys

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
    sys.exit(main(sys.argv[1:]))
