#!/usr/bin/env python

import sys

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
    sys.exit(main(sys.argv[1:]))
