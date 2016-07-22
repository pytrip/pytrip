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
import numpy as np

from pytrip.dos import DosCube
import pytriplib
import tests.test_base


class TestDos(unittest.TestCase):
    def setUp(self):
        testdir = tests.test_base.get_files()
        self.cube000 = os.path.join(testdir, "tst003001.dos")

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


if __name__ == '__main__':
    unittest.main()
