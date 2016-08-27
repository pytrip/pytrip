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

    basename = args[0].split(".")[0]
    output_folder = args[1]
    if os.path.exists(basename + ".ctx"):
        input_ctx_filename = basename + ".ctx"
    elif os.path.exists(basename + ".ctx.gz"):
        input_ctx_filename = basename + ".ctx.gz"
    else:
        print("There is no CTX file, script stop")
        exit()

    print("Convert CT images")
    c = CtxCube()
    c.read(input_ctx_filename)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    c.write_dicom(output_folder)

    if os.path.exists(basename + ".vdx"):
        print("Convert structures")
        v = VdxCube(c)
        v.read(basename + ".vdx")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        v.write_dicom(output_folder)
    print("Done")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
