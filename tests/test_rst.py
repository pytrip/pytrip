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
Test for raster.py
"""
import os
import unittest
import logging
import tempfile

import tests.base
from pytrip.raster import Rst
import pytrip.utils.rst2sobp

logger = logging.getLogger(__name__)


class TestRst(unittest.TestCase):
    """ Test the raster.py files
    """
    def setUp(self):
        """ Prepare files for tests
        """
        testdir = tests.base.get_files()
        self.rst000 = os.path.join(testdir, "tst003001.rst")
        logger.info("Testing .rst from path " + self.rst000)

    def test_read(self):
        """ Check if we are able to read a simple .rst file
        """
        r = Rst()
        r.read(self.rst000)
        self.assertEqual(r.submachines, '17')
        self.assertEqual(r.machines[0].points, 323)
        self.assertEqual(r.machines[0].energy, 120.2)
        self.assertEqual(r.machines[0].raster_points[0], [27.0, -24.0, 2844850.0])


class TestRst2Sobp(unittest.TestCase):
    """ Test the rst2sobp.py script
    """
    def setUp(self):
        """ Prepare files for tests
        """
        testdir = tests.base.get_files()
        self.rst000 = os.path.join(testdir, "tst003001.rst")
        logger.info("Testing rst2sobp.py using .rst from path " + self.rst000)

    def test_generate(self):
        """ Execute rst2sobp.py and make sure a non-empty file exists.
        """
        fd, outfile = tempfile.mkstemp()

        pytrip.utils.rst2sobp.main(args=[self.rst000, outfile])

        # check if destination file is not empty
        self.assertTrue(os.path.exists(outfile))
        self.assertGreater(os.path.getsize(outfile), 0)

        os.close(fd)  # Windows needs it
        os.remove(outfile)  # we need only temp filename, not the file


if __name__ == '__main__':
    unittest.main()
