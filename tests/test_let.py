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
import logging
import tempfile
import unittest

from pytrip.vdx import create_sphere
from pytrip.let import LETCube

import tests.base

logger = logging.getLogger(__name__)


class TestLet(unittest.TestCase):
    def setUp(self):
        testdir = tests.base.get_files()
        self.let001 = os.path.join(testdir, "tst003001.dosemlet.dos")
        self.vdx000 = os.path.join(testdir, "tst003000.vdx")

    def test_read(self):
        let = LETCube()
        let.read(self.let001)

        v = create_sphere(let, name="sph", center=[10, 10, 10], radius=8)
        self.assertIsNotNone(v)

        logger.info("Calculating DVH")
        result = let.calculate_lvh(v)
        self.assertIsNotNone(result)
        lvh, min_l, max_l, mean, area = result
        self.assertGreater(area, 2.0)
        self.assertEqual(len(lvh.shape), 1)
        self.assertEqual(lvh.shape[0], 3000)
        self.assertEqual(min_l, 0.0)
        self.assertEqual(max_l, 1.0)

        self.assertGreater(let.get_max(), 30.0)

        fd, outfile = tempfile.mkstemp()
        os.close(fd)  # Windows needs it
        os.remove(outfile)  # we need only temp filename, not the file
        let.write(outfile)
        hed_file = outfile + LETCube.header_file_extension
        dos_file = outfile + LETCube.data_file_extension
        self.assertTrue(os.path.exists(hed_file))
        self.assertTrue(os.path.exists(dos_file))
        logger.info("Checking if output file " + hed_file + " is not empty")
        self.assertGreater(os.path.getsize(hed_file), 1)
        logger.info("Checking if output file " + dos_file + " is not empty")
        self.assertGreater(os.path.getsize(dos_file), 1)
        os.remove(hed_file)
        os.remove(dos_file)

        # TODO fix it !
        # fd, outfile = tempfile.mkstemp()
        # os.close(fd)  # Windows needs it
        # os.remove(outfile)  # we need only temp filename, not the file
        # l.write_lvh_to_file(v, outfile)
        # lvh_file = outfile + ".dvh"
        # self.assertTrue(os.path.exists(lvh_file))
        # logger.info("Checking if output file " + lvh_file + " is not empty")
        # self.assertGreater(os.path.getsize(lvh_file), 1)
        # os.remove(lvh_file)


if __name__ == '__main__':
    unittest.main()
