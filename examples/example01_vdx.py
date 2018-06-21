#
#    Copyright (C) 2010-2017 PyTRiP98 Developers.
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
This example shows how to use contours to select volume of interests inside a CTX cube. The VOI is then manipulated.
"""

import logging
import pytrip as pt

# by default logging in PyTRiP98 is disabled, here we enable it to see it with basic INFO level
logging.basicConfig(level=logging.DEBUG)

# first define some paths and other important parameters
ctx_path = "/home/bassler/Projects/CTdata/TST000/TST000000.ctx"
vdx_path = "/home/bassler/Projects/CTdata/TST000/TST000000.vdx"
my_target_voi = "GTV"

# load CT cube
my_ctx = pt.CtxCube()
my_ctx.read(ctx_path)

# load VOIs
my_vdx = pt.VdxCube(my_ctx)  # my_vdx is the object which will hold all volumes of interest and the meta information
my_vdx.read(vdx_path)  # load the .vdx file
print(my_vdx.voi_names())  # show us all VOIs found in the .vdx file

# Select the requested VOI from the VdxCube object
target_voi = my_vdx.get_voi_by_name(my_target_voi)

# get_voi_cube() returns a DosCube() object, where all voxels inside the VOI holds the value 1000, and 0 elsewhere.
voi_cube = target_voi.get_voi_cube()

# Based on the retrieved DosCube() we calculate a three dimensional mask object,
# which assigns True to all voxels inside the Voi, and False elsewhere.
mask = (voi_cube.cube == 1000)

# "The mask object and the CTX cube have same dimensions (they are infact inherited from the same top level class).
# Therefore we can apply the mask cube to the ctx cube and work with the values.
# For instance we can set all HUs to zero within the Voi:
my_ctx.cube[mask] = 0
# or add 100 to all HUs of the voxels inside the mask:
# my_ctx.cube[mask] += 100

# save masked CT to the file in current directory
masked_ctx = "masked.ctx"
my_ctx.write(masked_ctx)
