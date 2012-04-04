from ctx2 import *
from dos2 import *
from vdx2 import *
import dicomhelper as dh

dcm = dh.read_dicom_folder("../../../DicomData/")
c = CtxCube()
c.read_dicom(dcm)

v = VdxCube("",c)
v.read_dicom(dcm)

c.write_dicom('../testfiles/')
