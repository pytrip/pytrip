"""
    This file is part of PyTRiP.

    PyTRiP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyTRiP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyTRiP.  If not, see <http://www.gnu.org/licenses/>
"""
# import os
# import numpy as np
# import pytrip.rst, pytrip.ddd, pytrip.field, time, \
#     pytrip.dos, pytrip.ctx, pytrip.vdx, pytrip.paths
# #import matplotlib.pyplot as plt
# from pytrip.res.point import *
#
# path = "/home/jato/Projects/TRiP/robustness/Carbon_Head/3mm"
# datafiles = os.listdir(path)
# i = 1
# with open(os.path.join(path, "results")) as fp:
#     dat = fp.read().split('\n')
# keys = dat[0:-1:7]
# dvh = dat[4:-1:7]
# qual = dat[5:-1:7]
# data = {}
# quality = {}
# for k, v, q in zip(keys, dvh, qual):
#     data[k] = v
#     quality[k] = q.split()[1]
# output = []
# x = []
# x2 = []
# y = []
# for item in datafiles:
#     if os.path.splitext(item)[1] == ".rst":
#         name = os.path.splitext(item)[0]
#         rs = rst.Rst()
#         rs.load_field(os.path.join(path, item))
#         avg = 0
#         i = 0
#         value = 0
#         # ~ for machine in rs.get_submachines():
#         # ~ i += 1
#         # ~ grid = machine.get_raster_grid()
#         # ~ if len(grid) == 1 or len(grid[0]) == 1:
#         # ~ continue
#         # ~ dat = np.gradient(grid)
#         # ~ norm = (dat[0]**2+dat[0]**2)**0.5
#         # ~ value += np.sum(norm)/np.sum(norm>0)/10000
#         # ~
#         i = 1
#
#         machine = rs.get_submachines()[0]
#         grid = machine.get_raster_grid()
#         dat = np.gradient(grid)
#         norm = (dat[0] ** 2 + dat[0] ** 2) ** 0.5
#         value = np.sum(norm) / np.sum(norm > 0) / 10000
#
#         machine = rs.get_submachines()[1]
#         grid = machine.get_raster_grid()
#         dat = np.gradient(grid)
#         norm = (dat[0] ** 2 + dat[0] ** 2) ** 0.5
#         value += np.sum(norm) / np.sum(norm > 0) / 10000
#
#         machine = rs.get_submachines()[-1]
#         grid = machine.get_raster_grid()
#         dat = np.gradient(grid)
#         norm = (dat[0] ** 2 + dat[0] ** 2) ** 0.5
#         value += np.sum(norm) / np.sum(norm > 0) / 10000
#
#         # ~ machine = rs.get_submachines()[-2]
#         # ~ grid = machine.get_raster_grid()
#         # ~ dat = np.gradient(grid)
#         # ~ norm = (dat[0]**2+dat[0]**2)**0.5
#         # ~ value += np.sum(norm)/np.sum(norm>0)/10000
#         # ~
#
#         y.append(1 - float(data[name]))
#         x.append(value / i)
#
#         x2.append(float(quality[name]))
# fig = plt.figure()
#
# ax = fig.add_subplot(2, 1, 1)
# plt.plot(x, y, '.')
#
# ax = fig.add_subplot(2, 1, 2)
# plt.plot(x2, y, '*')
#
# plt.show()
