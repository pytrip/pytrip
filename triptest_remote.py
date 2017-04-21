import pytrip as pt
import pytrip.tripexecuter2 as pte

import os
import logging


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

wdir = "/home/bassler/test"
patient_name = "TST000000"

ctx_path = os.path.join(wdir, patient_name + ".ctx")
vdx_path = os.path.join(wdir, patient_name + ".vdx")

c = pt.CtxCube()
c.read(ctx_path)

v = pt.VdxCube(c)
v.read(vdx_path)

print(v.get_voi_names())

plan = pte.Plan(basename=patient_name)

plan.ddd_dir = "/home/bassler/TRiP98/base/DATA/DDD/12C/RF3MM/*"
plan.spc_dir = "/home/bassler/TRiP98/base/DATA/SPC/12C/RF3MM/*"
plan.sis_path = "/home/bassler/TRiP98/base/DATA/SIS/12C.sis"
plan.hlut_path = "/home/bassler/TRiP98/base/DATA/HLUT/19990218.hlut"
plan.dedx_path = "/home/bassler/TRiP98/base/DATA/DEDX/20040607.dedx"
plan.working_dir = "/home/bassler/test/"  # working dir must exist.

# add the target voi to the plan
plan.voi_target = v.get_voi_by_name('CTV')

plan.rifi = 3.0
plan.bolus = 0.0
plan.offh2o = 1.873

# create a field and add it to the plan
field = pte.Field()
field.basename = patient_name
field.gantry = 10.0
field.couch = 90.0  # degrees
field.fwhm = 4.0  # spot size in [mm]
field.projectile = 'C'

print(field)
plan.fields.append(field)

# flags for what output should be generated
plan.want_phys_dose = True
plan.want_bio_dose = False
plan.want_dlet = True
plan.want_rst = False

# print(plan)
# print(plan.make_exec())

t = pte.Execute(c, v)
t.remote = True
t.servername = "titan.phys.au.dk"
t.username = "bassler"
t.remote_base_dir = "/home/bassler/test"
t.trip_bin_path = "/opt/aptg/TRiP98/bin/TRiP98"  # TODO: some issue with .bashrc in remote mode
# t._norun = True  # disable TRiP execution for now
t.execute(plan)

# results can be found in plan.dosecubes[] and plan.letcubes[], but they are also saved to working_dir
# d = plan.dosecubes[0]  # results stores here
# l = plan.letcubes[0]
