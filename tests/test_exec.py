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
TODO: documentation here.
"""
import logging
import os

import pytrip as pt
import pytrip.tripexecuter as pte

logger = logging.getLogger(__name__)


def test_exec(ctx_corename, vdx_filename, patient_name="tst003000"):
    """Prepare and execute a dry-run test using the Executer."""
    logger.info("Test norun TRiP98 execution")

    logger.debug("Load CtxCube {:s}".format(ctx_corename))
    c = pt.CtxCube()
    c.read(ctx_corename)

    logger.debug("Load VdxCube {:s}".format(vdx_filename))
    v = pt.VdxCube(c)
    v.read(vdx_filename)

    print(v.voi_names())

    projectile = pte.Projectile('H')
    kernel = pte.KernelModel(projectile)
    kernel.ddd_path = "/opt/TRiP98/base/DATA/DDD/12C/RF3MM/*"
    kernel.spc_path = "/opt/TRiP98/base/DATA/SPC/12C/RF3MM/*"
    kernel.sis_path = "/opt/TRiP98/base/DATA/SIS/12C.sis"
    kernel.rifi_thickness = 3.0
    plan = pte.Plan(default_kernel=kernel, basename=patient_name)
    assert plan is not None

    plan.hlut_path = "/opt/TRiP98/base/DATA/HLUT/19990218.hlut"
    plan.dedx_path = "/opt/TRiP98/base/DATA/DEDX/20040607.dedx"
    plan.working_dir = "."  # working dir must exist.

    # add the target voi to the plan
    plan.voi_target = v.get_voi_by_name('target')

    plan.bolus = 0.0
    plan.offh2o = 1.873

    # create a field and add it to the plan
    field = pte.Field(kernel)
    assert field is not None
    field.basename = patient_name
    field.gantry = 10.0
    field.couch = 90.0  # degrees
    field.fwhm = 4.0  # spot size in [mm]

    plan.fields.append(field)

    # flags for what output should be generated
    plan.want_phys_dose = True
    plan.want_bio_dose = False
    plan.want_dlet = True
    plan.want_rst = False

    t = pte.Execute(c, v)
    assert t is not None
    t.trip_bin_path = 'tests/res/dummy_TRiP98/TRiP98'
    if os.name != 'nt':  # skip running fake TRiP98 on Windows as it is not supported there
        t.execute(plan=plan, run=False)  # setup and make a dry-run, since TRiP98 is not installed.

    executer_str = str(t)
    assert len(executer_str) > 1
    # No results will be generated since, TRiP98 is not installed in test environment.
