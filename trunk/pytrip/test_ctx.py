from ctx2 import *
from dos2 import *
from vdx2 import *
import dicomhelper as dh

dcm = dh.read_dicom_folder("../../../DicomData/Dicompyler/")
c = CtxCube()
c.read_dicom(dcm)
print c.cube[50][100][100]
d=c+5

print d.cube[50][100][100]
