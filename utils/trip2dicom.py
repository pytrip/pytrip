#!/usr/bin/env python

from pytrip import *
import os
import sys
print 
print "\ttrip2dicom is a part of pytrip which was developed by \n\tNiels Bassler (bassler@phys.au.dk) and \n\tJakob Toftegaard (jakob.toftegaard@gmail.com)"
print
if len(sys.argv) != 3:
    print "\tusage: trip2dicom.py headerfile output_folder"
    print "\ttrip2dicom.py tripfile.hed dicomfolder/"
    print 
    exit()
    
basename = sys.argv[1].split(".")[0]
output_folder = sys.argv[2]
if os.path.exists(basename + ".ctx"):
    print "Convert CT images"
    c = CtxCube()
    c.read(basename + ".ctx")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    c.write_dicom(output_folder)
else:
    print "There is no CTX file, script stop"
    exit()
if os.path.exists(basename + ".vdx"):
    print "Convert structures"
    v = VdxCube(c)
    v.read(basename + ".vdx")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    v.write_dicom(output_folder)
print "Done"


