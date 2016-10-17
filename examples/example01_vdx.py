import pytrip as pt
from pytrip.utils import cubeslice

# first define some paths and other important parameters
patient_name = "TST000000"
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

# next pick a the proper VOI which we want to plan on
target_voi = my_vdx.get_voi_by_name(my_target_voi)  # Select the requested VOI from the VdxCube object

# dos cube representing VOI, it is filled with zeros outside the volume and with 1000 inside
voi_cube = target_voi.get_voi_cube()

# mask object, same shape as dos and voi cube, true inside volume, false outside
mask = (voi_cube.cube == 1000)

# make a masked CT - set Hounsfield units to zeros inside the volume
my_ctx.cube[mask] = 0

# save masked CT to the file in current directory
masked_ctx = "masked.ctx"
my_ctx.write(masked_ctx)

# optionally save one slice from original and masked CTs to the png images in current directory
# it is ugly usage of command-line cubeslice tool, we will improve it with time
cubeslice.main(["--ct", ctx_path, "-f", "49", "-t", "50", "-o", "."])
cubeslice.main(["--ct", masked_ctx, "-f", "49", "-t", "50", "-o", "."])
