#!/usr/bin/env python

import sys

from pytrip import CtxCube, dicomhelper

basename = sys.argv[2]

# import DICOM
dcm = dicomhelper.read_dicom_folder(sys.argv[1])
c = CtxCube()
c.read_dicom(dcm)

c.write_trip_header(basename+".hed")
c.write_trip_data(basename+".ctx")

#v = VdxCube("",c)
#v.read_dicom(dcm)
#v.write_to_trip(basename+".vdx")

exit()
