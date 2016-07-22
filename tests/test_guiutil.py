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


from pytrip.ctx import CtxCube
from pytrip.vdx import VdxCube
from pytrip.guiutil import PlotUtil

import tests.test_base


class TestCtx(unittest.TestCase):
    def setUp(self):
        testdir = tests.test_base.get_files()
        self.cube000 = os.path.join(testdir, "tst003000.ctx")
        self.vdx000 = os.path.join(testdir, "tst003000.vdx")

    def test_read(self):
        c = CtxCube()
        c.read(self.cube000)
        v = VdxCube(c)
        v.read_vdx(self.vdx000)
        g = PlotUtil()
        g.set_ct(c)
#        g.add_voi(v.get_voi_by_name("ptv"))
#        g.plot(81)
#        g.plot(82)

if __name__ == '__main__':
    unittest.main()
