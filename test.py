import pytrip as pt

ctxf = "../shieldhit/res/TST001/tst001000.ctx"
#ctxf = "foobar.ctx"

c = pt.CtxCube()
c.read(ctxf)

dcm_list = c.create_dicom()

#for i in range(len(dcm_list)):
#    print("{:d}     {}".format(i,dcm_list[i].SeriesInstanceUID))

c.write_dicom("./dcm4")

#print(c.data[0].StudyDate)
#print(c.data[0].SeriesInstanceUID)
