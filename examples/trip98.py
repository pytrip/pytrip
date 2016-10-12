import pytrip as pt
import pytrip.tripexecuter as pte

# first define some paths and other important paramters
patient_name = "TST000000"
ctx_path = "/home/bassler/Projects/shieldhit/res/TST000/TST000000.ctx"
vdx_path = "/home/bassler/Projects/shieldhit/res/TST000/TST000000.vdx"
voi_name = "GTV"
working_directory = "/home/bassler/Projects/shieldhit/res/TST000/"

ddd_dir = "/home/bassler/TRiP98/base/DATA/DDD/12C/RF3MM"
spc_dir = "/home/bassler/TRiP98/base/DATA/SPC/12C/RF3MM"
sis_path = "/home/bassler/TRiP98/base/DATA/SIS/12C.sis"
hlut_path = "/home/bassler/TRiP98/base/DATA/HLUT/19990218.hlut"
dedx_path = "/home/bassler/TRiP98/base/DATA/DEDX/20040607.dedx"

my_couch_angle = 90.0
my_gantry_angle = 10.0
my_target_voi = "GTV"  # the name must exist in the .vdx file
my_projectile = "C"

# load CT cube
my_ctx = pt.CtxCube()
my_ctx.read(ctx_path)

# load VOIs
my_vdx = pt.VdxCube("", my_ctx)  # my_vdx is the object which will hold all volumes of interest and the meta information
my_vdx.read(vdx_path)  # load the .vdx file
print(my_vdx.get_voi_names())  # show us all VOIs found in the .vdx file

# next pick a the proper VOI which we want to plan on
target_voi_temp = my_vdx.get_voi_by_name(my_target_voi)  # Select the requested VOI from the VdxCube object
target_voi = pte.Voi("GTV_VOI", target_voi_temp)  # for technical reasons the voi must be cast into a new Voi object

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

# Remote access currently broken, see issue #94
# https://github.com/pytrip/pytrip/issues/94
# my_plan.set_remote_state(True)
# my_plan.set_server("titan.phys.au.dk")
# my_plan.set_username("xxxxxxxxx")
# my_plan.set_password("xxxxxxxxx")

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
my_trip.execute(my_plan)
