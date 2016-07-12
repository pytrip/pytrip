# from pytrip.paths import *
# import time
# from pytrip.res.point import *
# from pytrip.ctx import *
# from pytrip.vdx import *
# from pytrip.dos import *
# import os
# import gc
# import pytrip.dicomhelper as dh
# import numpy
# import threading
# import matplotlib.pyplot as plt
#
# dicomFolder = '/home/jato/Projects/Oscar/WEPL calculation/data/'
# structureFolder = '/home/jato/Projects/Oscar/WEPL calculation/data/structure'
# gantry = range(180,360,10)
# couch = [0]
#
#
# dcm = dh.read_dicom_folder(structureFolder)
#
# path = os.path.join(dicomFolder,'00')
# c = CtxCube()
# dcm = dh.read_dicom_folder(path)
# c.read_dicom(dcm)
#
# dcm = dh.read_dicom_folder(structureFolder)
# v = VdxCube("",c)
# v.read_dicom(dcm)
#
# gc.collect()
# voi = v.get_voi_by_name("Tumor")
# d = voi.get_voi_cube()
# d.cube = np.array(d.cube,dtype=np.float32)
# voi_cube = DensityProjections(d)
#
# for phase in ['00','10','20','30','40','50','60','70','80','90']:
#     print phase
#     path = os.path.join(dicomFolder,phase)
#     c = CtxCube()
#     dcm = dh.read_dicom_folder(path)
#     c.read_dicom(dcm)
#     p = DensityCube(c)
#     projection = DensityProjections(p)
#     for g in gantry:
#         for c in couch:
#             data,start,basis = projection.calculate_projection(voi,g,c,0,1.0)
#             voi_proj,t1,t2 = voi_cube.calculate_projection(voi,g,c,1,1.0)
#             dmap = data*(voi_proj>0.0)
#             np.save("output/%s-%d-%d"%(phase,g,c),dmap)
#
#
#
#
