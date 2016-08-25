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

from pytrip.vdx import VdxCube

from pytrip.let import LETCube

import tests.test_base


class TestLet(unittest.TestCase):
    def setUp(self):
        testdir = tests.test_base.get_files()
        self.let001 = os.path.join(testdir, "tst003001.dosemlet.dos")
        self.vdx000 = os.path.join(testdir, "tst003000.vdx")

    def test_read(self):
        l = LETCube()
        l.read_trip_data_file(self.let001)
        v = VdxCube("")
        v.import_vdx(self.vdx000)
        # TODO fix loading VOI data
        # tumor = v.get_voi_by_name("Tumor Bed")
        # lvh = l.calculate_lvh(tumor)
        # plt.plot(lvh[0], lvh[1])
        # plt.show()


if __name__ == '__main__':
    unittest.main()
