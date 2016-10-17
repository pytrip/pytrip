import pytrip as pt

# first define some paths and other important parameters
ctx_path = "/home/bassler/Projects/CTdata/TST000/TST000000.ctx"
vdx_path = "/home/bassler/Projects/CTdata/TST000/TST000000.vdx"
my_target_voi = "GTV"

# load CT cube
my_ctx = pt.CtxCube()
my_ctx.read(ctx_path)

# load VOIs
my_vdx = pt.VdxCube("", my_ctx)  # my_vdx is the object which will hold all volumes of interest and the meta information
my_vdx.read(vdx_path)  # load the .vdx file
print(my_vdx.get_voi_names())  # show us all VOIs found in the .vdx file

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

# save masked CT to the file in current directory
masked_ctx = "masked.ctx"
my_ctx.write(masked_ctx)
