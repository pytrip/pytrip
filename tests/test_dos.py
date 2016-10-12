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
import os
import unittest
import logging
import tempfile
import shutil
import numpy as np

from pytrip.dos import DosCube
from pytrip.vdx import create_sphere
import pytriplib
import tests.test_base

logger = logging.getLogger(__name__)


class TestDos(unittest.TestCase):
    def setUp(self):
        testdir = tests.test_base.get_files()
        self.cube000 = os.path.join(testdir, "tst003001.dos")
        logger.info("Testing cube from path " + self.cube000)

    def test_read(self):
        c = DosCube()
        c.read(self.cube000)
        self.assertEqual(c.cube.shape[0], 300)
        self.assertEqual(c.cube.shape[1], 512)
        self.assertEqual(c.cube.shape[2], 512)

        # test method from C extension
        dose_center = pytriplib.calculate_dose_center(np.array(c.cube))
        self.assertEqual(dose_center.shape[0], 3)
        self.assertGreater(dose_center[0], 0.0)
        self.assertGreater(dose_center[1], 0.0)
        self.assertGreater(dose_center[2], 0.0)

    def test_dvh(self):
        c = DosCube()
        c.read(self.cube000)
        v = create_sphere(c, name="sph", center=[10, 10, 10], radius=8)
        self.assertIsNotNone(v)

        logger.info("Calculating DVH")
        result = c.calculate_dvh(v)
        self.assertIsNotNone(result)
        dvh, min_dose, max_dose, mean, area = result
        self.assertGreater(area, 2.0)
        self.assertEqual(len(dvh.shape), 2)
        self.assertEqual(dvh.shape[1], 2)
        self.assertEqual(dvh.shape[0], 1500)
        self.assertEqual(min_dose, 0.0)
        self.assertEqual(max_dose, 0.001)

    def test_dicom_plan(self):
        c = DosCube()
        c.read(self.cube000)

        dp = c.create_dicom_plan()
        self.assertIsNotNone(dp)

        d = c.create_dicom()
        self.assertIsNotNone(d)

    def test_write_dicom(self):
        c = DosCube()
        c.read(self.cube000)

        outdir = tempfile.mkdtemp()
        c.write_dicom(outdir)
        self.assertTrue(os.path.exists(os.path.join(outdir, "rtdose.dcm")))
        self.assertTrue(os.path.exists(os.path.join(outdir, "rtplan.dcm")))
        self.assertGreater(os.path.getsize(os.path.join(outdir, "rtdose.dcm")), 0)
        self.assertGreater(os.path.getsize(os.path.join(outdir, "rtplan.dcm")), 0)
        shutil.rmtree(outdir)

    def test_write(self):
        c = DosCube()
        c.read(self.cube000)

        fd, outfile = tempfile.mkstemp()
        os.close(fd)  # Windows needs it
        os.remove(outfile)  # we need only temp filename, not the file
        c.write(outfile)
        hed_file = outfile + ".hed"
        dos_file = outfile + ".dos"
        self.assertTrue(os.path.exists(hed_file))
        self.assertTrue(os.path.exists(dos_file))
        logger.info("Checking if output file " + hed_file + " is not empty")
        self.assertGreater(os.path.getsize(hed_file), 1)
        logger.info("Checking if output file " + dos_file + " is not empty")
        self.assertGreater(os.path.getsize(dos_file), 1)
        os.remove(hed_file)
        os.remove(dos_file)

if __name__ == '__main__':
    unittest.main()
