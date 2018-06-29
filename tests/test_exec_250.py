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
import os
import unittest
import logging

import pytrip as pt
import pytrip.tripexecuter as pte

import tests.base

logger = logging.getLogger(__name__)


class TestLocalExec250(unittest.TestCase):
    """
    Tests for pytrip.tripexecuter, using new API slated for 2.5.0
    """
    def setUp(self):
        """ Prepare test environment.
        """
        testdir = tests.base.get_files()

        self.patient_name = "tst003000"

        self.ctx_path = os.path.join(testdir, self.patient_name + '.ctx')
        self.vdx_path = os.path.join(testdir, self.patient_name + '.vdx')
        self.trip_path = os.path.join(testdir, "TRiP98")

    def test_exec(self):
        """ Prepare and execute a dry-run test using the Executer.
        """
        logger.info("Test norun TRiP98 execution")

        logger.debug("Load CtxCube {:s}".format(self.ctx_path))
        c = pt.CtxCube()
        c.read(self.ctx_path)

        logger.debug("Load VdxCube {:s}".format(self.vdx_path))
        v = pt.VdxCube(c)
        v.read(self.vdx_path)

        print(v.voi_names())

        kernel = pte.KernelModel("C12 Ions RiFi 3MM")
        kernel.projectile = pte.Projectile("C")
        kernel.ddd_path = "/opt/TRiP98/base/DATA/DDD/12C/RF3MM/*"
        kernel.spc_path = "/opt/TRiP98/base/DATA/SPC/12C/RF3MM/*"
        kernel.sis_path = "/opt/TRiP98/base/DATA/SIS/12C.sis"
        kernel.rifi_thickness = 3.0
        kernel.rifi_name = "GSI RF3MM"
        kernel.comment = "C-12 Ions with a 3 mm 1-D Ripple Filter from GSI"

        plan = pte.Plan(basename=self.patient_name, kernels=(kernel, ))
        self.assertIsNotNone(plan)

        plan.hlut_path = "/opt/TRiP98/base/DATA/HLUT/19990218.hlut"
        plan.dedx_path = "/opt/TRiP98/base/DATA/DEDX/20040607.dedx"
        plan.working_dir = "."  # working dir must exist.

        # add the target voi to the plan
        plan.voi_target = v.get_voi_by_name('target')

        plan.bolus = 0.0
        plan.offh2o = 1.873

        # create a field and add it to the plan
        field = pte.Field(kernel=kernel)
        self.assertIsNotNone(field)
        field.basename = self.patient_name
        field.gantry = 10.0
        field.couch = 90.0  # degrees
        field.fwhm = 4.0  # spot size in [mm]
        field.projectile = 'C'

        plan.fields.append(field)

        # flags for what output should be generated
        plan.want_phys_dose = True
        plan.want_bio_dose = False
        plan.want_dlet = True
        plan.want_rst = False

        t = pte.Execute(c, v)
        self.assertIsNotNone(t)
        t.trip_bin_path = self.trip_path
        print(self.trip_path)
        if os.name != 'nt':  # skip running fake TRiP98 on Windows as it is not supported there
            t.execute(plan, False)  # setup and make a dry-run, since TRiP98 is not installed.

        executer_str = str(t)
        self.assertGreater(len(executer_str), 1)
        # No results will be generated since, TRiP98 is not installed in test environment.


if __name__ == '__main__':
    unittest.main()
