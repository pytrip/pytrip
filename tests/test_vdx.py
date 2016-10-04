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

from pytrip.ctx import CtxCube
from pytrip.vdx import VdxCube
import tests.test_base


logger = logging.getLogger(__name__)

class TestVdx(unittest.TestCase):
    def setUp(self):
        testdir = tests.test_base.get_files()
        self.cube000 = os.path.join(testdir, "tst003000")
        self.vdx = os.path.join(testdir, "tst003000.vdx")

    def test_read_with_ct(self):
        logger.info("Testing cube from path " + self.cube000)
        c = CtxCube()
        c.read(self.cube000)
        v = VdxCube(c, "")
        logger.info("Testing vdx from path " + self.vdx)
        v.read(self.vdx)

        logger.info("Checking len of get_voi_names ")
        self.assertGreater(len(v.get_voi_names()), 1)

        logger.info("Checking str method ")
        self.assertGreater(len(str(v)), 1)

    def test_read_solo(self):
        v = VdxCube("")
        v.read(self.vdx)

if __name__ == '__main__':
    unittest.main()
