#
#    Copyright (C) 2010-2016 PyTRiP98 Developers.
#
#    This file is part of PyTRiP98.
#
#    PyTRiP98 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyTRiP98 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyTRiP98.  If not, see <http://www.gnu.org/licenses/>.
#
"""
This example demonstrates how to load a CT cube in Voxelplan format, and the associated contours.
Then a plan is prepared and optimized using TRiP98.
The resulting dose plan is then stored in <some_dir>.
"""

import pytrip as pt
import pytrip.tripexecuter as pte

# first define some paths and other important parameters
patient_name = "TST000000"
ctx_path = "/home/bassler/Projects/CTdata/TST000/TST000000.ctx"
vdx_path = "/home/bassler/Projects/CTdata/TST000/TST000000.vdx"
voi_name = "GTV"
working_directory = "/home/bassler/Projects/CTdata/TST000/"

ddd_dir = "/home/bassler/TRiP98/base/DATA/DDD/12C/RF3MM"
spc_dir = "/home/bassler/TRiP98/base/DATA/SPC/12C/RF3MM"
sis_path = "/home/bassler/TRiP98/base/DATA/SIS/12C.sis"
hlut_path = "/home/bassler/TRiP98/base/DATA/HLUT/19990218.hlut"
dedx_path = "/home/bassler/TRiP98/base/DATA/DEDX/20040607.dedx"

my_couch_angle = 90.0
my_gantry_angle = 10.0
my_target_voi = "GTV"  # the name must exist in the .vdx file
my_projectile = "C"  # carbon ion

# load CT cube
my_ctx = pt.CtxCube()
my_ctx.read(ctx_path)

# load VOIs
my_vdx = pt.VdxCube("", my_ctx)  # my_vdx is the object which will hold all volumes of interest and the meta information
my_vdx.read(vdx_path)  # load the .vdx file
print(my_vdx.get_voi_names())  # show us all VOIs found in the .vdx file

# next pick a the proper VOI (from the VdxCube object) which we want to plan on
target_voi_temp = my_vdx.get_voi_by_name(my_target_voi)

# for technical reasons the voi must be cast into a new Voi object
# we are working on a cleaner Voi class implementation to avoid it
target_voi = pte.Voi("GTV_VOI", target_voi_temp)

# Next, setup a plan. We may initialize it with a new name.
# The name must be identical to the base name of the file, else we will have crash
my_plan = pte.TripPlan(name=patient_name)

# set working directory, output will go there
my_plan.set_working_dir(working_directory)
my_plan.set_ddd_folder(ddd_dir)
my_plan.set_spc_folder(spc_dir)
my_plan.set_dedx_file(dedx_path)
my_plan.set_hlut_file(hlut_path)
my_plan.set_sis_file(sis_path)

# To enable remote access to trip, uncomment and eddit the following:
# my_plan.set_remote_state(True)
# my_plan.set_server("titan.phys.au.dk")  # location of remote TRiP98 installation. Needs SSH access.
# my_plan.set_username("xxxxxxxxx")
# my_plan.set_password("xxxxxxxxx")  # to login using SSH-keys, leave this commented out.

# add target VOI to the plan
my_plan.add_voi(target_voi)
my_plan.get_vois()[0].target = True  # make TRiP98 aware of that this VOI is the target.

# Finally we need to add a field to the plan
# add default field, carbon ions
my_field = pte.Field("Field 1")
my_field.set_projectile(my_projectile)  # set the projectile
my_field.set_couch(my_couch_angle)
my_field.set_gantry(my_gantry_angle)

my_plan.add_field(my_field)

# the next line is needed to correctly set offset between VOI and CT
ct_images = pt.CTImages(my_ctx)

# run TRiP98 optimisation
my_trip = pte.TripExecuter(ct_images.get_modified_images(my_plan))
# TRiP98 will then run the plan and generate the requested dose plan.
# The dose plan is stored in the working directory, and must then be loaded by the user for further processing.
# for local execution, we assume TRiP98 binary is present in PATH env. variable
my_trip.execute(my_plan)
