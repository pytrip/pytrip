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
This example demonstrates how to load a CT cube in Voxelplan format, and the associated contours.
Then a plan is prepared and optimized using TRiP98.
"""
import os
import logging

import pytrip as pt
import pytrip.tripexecuter as pte

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # give some output on what is going on.

# Fist we specify the directory where all our files are:
wdir = "/home/bassler/test"

# In TRiP, the patient "TST000" would typically carry the filename "TST000000"
patient_name = "TST000000"

# so we can construc the paths to the CTX and VDX files like this:
ctx_path = os.path.join(wdir, patient_name + ".ctx")
vdx_path = os.path.join(wdir, patient_name + ".vdx")

# Next we load the CT cube:
c = pt.CtxCube()
c.read(ctx_path)

# And load the contours
v = pt.VdxCube(c)
v.read(vdx_path)

# we may print all contours found in the Vdx file, if we want to
print(v.get_voi_names())

# Ok, we have the Contours and the CT cube ready. Next we must prepare a plan.
# We may choose any basename for the patient. All output files will be named using
# this basename.
plan = pte.Plan(basename=patient_name)

# We need to specify where the kernel files can be found. The location may depend on the ion we
# wnat to treat with. This example is for carbon ions:
plan.ddd_dir = "/home/bassler/TRiP98/base/DATA/DDD/12C/RF3MM/*"
plan.spc_dir = "/home/bassler/TRiP98/base/DATA/SPC/12C/RF3MM/*"
plan.sis_path = "/home/bassler/TRiP98/base/DATA/SIS/12C.sis"
plan.hlut_path = "/home/bassler/TRiP98/base/DATA/HLUT/19990218.hlut"
plan.dedx_path = "/home/bassler/TRiP98/base/DATA/DEDX/20040607.dedx"
plan.working_dir = "/home/bassler/test/"  # working dir must exist.

# Set the plan target to the voi called "CTV"
plan.voi_target = v.get_voi_by_name('CTV')

# some optional parameters (if not set, they will all be zero by default)
plan.rifi = 3.0  # 3 mm ripple filter. (This is only for documentaiton, it will not affect the dose optimization.)
plan.bolus = 0.0  # No bolus is applied here. Set this to some value, if you are to optimize very shallow tumours.
plan.offh2o = 1.873  # Some offset mimicing the monitoring ionization chambers and exit window of the beam nozzle.

# Next we need to specify at least one field, and add that field to the plan.
field = pte.Field()
field.basename = patient_name  # This name will be used for output filenames, if any field specific output is saved.
field.gantry = 10.0  # degrees
field.couch = 90.0  # degrees
field.fwhm = 4.0  # spot size in [mm]
field.projectile = 'C'

print(field)  # We can print all parameters of this field, for checking.
plan.fields.append(field)  # attach field to plan. You may attach multiple fields.

# Next, set the flags for what output should be generated, when the plan has completed.
plan.want_phys_dose = True  # We want a physical dose cube, "TST000000.dos"
plan.want_bio_dose = False  # No biological cube (Dose * RBE)
plan.want_dlet = True  # We want to have the dose-averaged LET cube
plan.want_rst = False  # Print the raster scan files (.rst) for all fields.

# print(plan)  # this will print all plan parameters

te = pte.Execute(c, v)  # get the executer object, based on the given Ctx and Vdx cube.

# in the case that TRiP98 is not installed locally, you may have to enable remote execution:
# te.remote = True
# te.servername = "titan.phys.au.dk"
# te.username = "bassler"
# te.password = "xxxxxxxx"  # you can set a password, but this is strongly discouraged. Better to exchange SSH keys!
# te.remote_base_dir = "/home/bassler/test"
# depending on the remote .bashrc_profile setup, it may be needed to specify the full path
# for the remote TRiP installation. On some systems the $PATH is set, so this line can be omitted,
# or shortened to just "TRiP98"
# te.trip_bin_path = "/opt/aptg/TRiP98/bin/TRiP98"

te.execute(plan)  # this will run TRiP
# te.execute(plan, False)  # set to False, if TRiP98 should not be executed. Good for testing.

# requested results can be found in
# plan.dosecubes[]
# and
# plan.letcubes[]
# and they are also saved to working_dir
