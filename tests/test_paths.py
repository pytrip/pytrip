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
# from pytrip.paths import *
# import time
# from pytrip.res.point import *
# from pytrip.ctx import *
# from pytrip.vdx import *
# from pytrip.dos import *
# import pytrip.dicomhelper as dh
# import numpy
# import threading
# #import matplotlib.pyplot as plt

# dcm = dh.read_dicom_folder(
# "../../../DicomData/Til Jacob T anonymiseret/Anonym2")
# c = CtxCube()
# c.read_dicom(dcm)
# ~ c.write_trip_data("../../../DicomData/test/test.ctx")
# ~ c.write_trip_header("../../../DicomData/test/test.hed")

# v = VdxCube("",c)
# v.read_dicom(dcm)
# #~ v.write_to_trip("../../../DicomData/test/test.vdx")
# voi = v.get_voi_by_name("ptv 3mm")
# #~ d.load_from_structure(voi,1000)
# voi.create_point_tree()
# #~ exit()
# p = DensityCube(c)
# projection = DensityProjections(p)
# #~ print(projection.calculate_angle_quality(voi,110,0)
# grid = projection.calculate_quality_grid(voi,range(0,360,2),range(-90,90,2))
# numpy.save("../plotdata",grid)
# exit()

# dcm = dh.read_dicom_folder(
# "../../../DicomData/Til Jacob T anonymiseret/Anonymiseret")
# c = CtxCube()
# c.read_dicom(dcm)
# #~ c.write_trip_data("../../../DicomData/test/test.ctx")
# #~ c.write_trip_header("../../../DicomData/test/test.hed")
#
# v = VdxCube("",c)
# v.read_dicom(dcm)
# #~ v.write_to_trip("../../../DicomData/test/test.vdx")
# voi = v.get_voi_by_name("ctv")
# #~ d.load_from_structure(voi,1000)
# voi.create_point_tree()
# #~ exit()
# p = DensityCube(c)
# projection = DensityProjections(p)
# #~ print(projection.calculate_angle_quality(voi,110,0)
# grid2 = projection.calculate_quality_grid(
# voi,range(0,360,10),range(-90,90,10))
# #~ numpy.save("../plotdata",grid)
#
# plt.imshow(grid/grid2)
# plt.colorbar()
# plt.show()
#
#
#
# #~ print(projection.calculate_angle_quality(voi,130,30)
# #~ for angle in angles:
# gantry = 90
# couch = 0
# print(projection.calculate_angle_quality(voi,gantry,couch))
# exit()
# #~ gantry,couch = angles_from_trip(gantry,couch)
# data,start,basis = projection.calculate_projection(voi,gantry,couch)
# #~ print(voi.calculate_center()
# #~ print(start
# contour = voi.get_2d_projection_on_basis(basis,start)
# #~ gradient = numpy.gradient(data)
# #~ data = (gradient[0]**2+gradient[1]**2)**0.5
# plt.imshow(data>0)
# plt.plot(contour[:,1]/c.pixel_size,contour[:,0]/c.pixel_size,'r')
# plt.colorbar()
# plt.show()
# exit()
#
# #~ print(
# gantry = 100
# couch =  -60
# print(gantry,couch)
# print(projection.calculate_angle_quality(voi,gantry,couch))
#
#
# gantry = 270
# couch =  70
# print(gantry,couch)
# print(projection.calculate_angle_quality(voi,gantry,couch))
#
# gantry = -110
# couch =  10
# print(gantry,couch)
# print(projection.calculate_angle_quality(voi,gantry,couch))
#
# gantry = 270
# couch =  50
# print(gantry,couch)
# print(projection.calculate_angle_quality(voi,gantry,couch))
#
# gantry = 180
# couch =  0
# print(gantry,couch)
# print(projection.calculate_angle_quality(voi,gantry,couch))
#
# gantry = -30
# couch =  60
# print(gantry,couch)
# print(projection.calculate_angle_quality(voi,gantry,couch))
#
# gantry = 100
# couch =  40
# print(gantry,couch)
# print(projection.calculate_angle_quality(voi,gantry,couch))
#
# gantry = 110
# couch =  10
# print(gantry,couch)
# print(projection.calculate_angle_quality(voi,gantry,couch))
#
# gantry = 0
# couch =  -90
# print(gantry,couch)
# print(projection.calculate_angle_quality(voi,gantry,couch))
#
#
# angles = [[100,-60],[270,70],[-110,10],[270,50],[180,0],
# [-30,60],[100,40],[110,10],[0,-90]]
# #~ angles = [[180,90]]
# for angle in angles:
#     data,start,basis = projection.calculate_projection(voi,angle[0],angle[1])
#     contour = voi.get_2d_projection_on_basis(basis,start)
#     gradient = numpy.gradient(data)
#     data = (gradient[0]**2+gradient[1]**2)**0.5
#     #~ print(start
#     #~ print(basis
#     plt.imshow(data,vmax=5.0)
#     plt.colorbar()
#     plt.plot(contour[:,1]/c.pixel_size,contour[:,0]/c.pixel_size,'r')
#     plt.show()
#
