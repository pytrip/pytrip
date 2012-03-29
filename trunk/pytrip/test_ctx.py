from ctx2 import *
import dicomhelper as dh
dcm = dh.import_dicom_folder("../../../DicomData/testdata/testdata/")
print "import"
v = CtxCube()
d = DosCube()
v.read_dicom(dcm)
d.read_dicom(dcm)
print "read"
v.write_dicom("../testfiles/")

