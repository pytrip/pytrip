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
# phases = ['00','10','20','30','40','50','60','70','80','90']
#
# gantry = range(180,360,10)
# couch = [0]
#
# output = np.zeros((len(gantry),len(couch)))
#
# for id1,g in enumerate(gantry):
#     for id2,c in enumerate(couch):
#         for i,phase in enumerate(phases):
#             d = np.load("output/%s-%d-%d.npy"%(phase,g,c))
#             if i == 0:
#                 data = np.zeros((10,d.shape[0],d.shape[1]))
#             data[i,:,:] = d
#         std = np.std(data,0)
#         output[id1,id2] = mean(std)
# plt.imshow(output)
# plt.show()
# np.savetxt("result.txt",output)
