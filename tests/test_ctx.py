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
import gzip
import hashlib
import os
import tempfile
import unittest
import logging

import tests.base
from pytrip.ctx import CtxCube
from pytrip.error import FileNotFound
from pytrip.util import TRiP98FileLocator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestCtx(unittest.TestCase):
    def setUp(self):
        testdir = tests.base.get_files()
        self.cube000 = os.path.join(testdir, "tst003000")

    def test_read(self):
        c = CtxCube()
        c.read(self.cube000)

    def read_and_write_cube(self, path):

        logger.info("Testing cube from path " + path)

        # read original cube and calculate hashsum
        c = CtxCube()
        c.read(path)

        # get path to the cube data file, extracting it from a partial path
        data_file_path = TRiP98FileLocator(self.cube000, CtxCube).datafile

        # get the hashsum
        if data_file_path.endswith(".gz"):
            f = gzip.open(data_file_path)
        else:
            f = open(data_file_path, 'rb')
        original_md5 = hashlib.md5(f.read()).hexdigest()
        f.close()

        # calculate temporary filename
        fd, outfile = tempfile.mkstemp()
        os.close(fd)
        os.remove(outfile)          # we need only random name, not a descriptor
        logger.debug("Generated random file name " + outfile)

        # save cube and calculate hashsum
        saved_header_path, saved_cubedata_path = c.write(outfile)   # this will write outfile+".ctx"  and outfile+".hed"

        # check if generated files exists
        self.assertTrue(os.path.exists(saved_header_path))
        self.assertTrue(os.path.exists(saved_cubedata_path))

        # get checksum
        f = open(saved_cubedata_path, 'rb')
        generated_md5 = hashlib.md5(f.read()).hexdigest()
        f.close()
        logger.debug("Removing " + saved_cubedata_path)
        os.remove(saved_cubedata_path)
        logger.debug("Removing " + saved_header_path)
        os.remove(saved_header_path)
        # compare checksums
        self.assertEqual(original_md5, generated_md5)

    def test_write(self):
        possible_names = [self.cube000,
                          self.cube000 + ".ctx", self.cube000 + ".hed",
                          self.cube000 + ".CTX", self.cube000 + ".HED",
                          self.cube000 + ".hed.gz", self.cube000 + ".ctx.gz"]

        for name in possible_names:
            self.read_and_write_cube(name)

    def test_problems_when_reading(self):
        # check malformed filename
        with self.assertRaises(FileNotFound) as e:
            logger.info("Catching {:s}".format(str(e)))
            self.read_and_write_cube(self.cube000[2:-1])

        # check exception if filename is without dot
        with self.assertRaises(FileNotFound) as e:
            logger.info("Catching {:s}".format(str(e)))
            self.read_and_write_cube(self.cube000 + "hed")

        # check opening wrong filetype (file self.cube000 + ".vdx" exists !)
        with self.assertRaises(FileNotFound) as e:
            logger.info("Catching {:s}".format(str(e)))
            self.read_and_write_cube(self.cube000 + ".vdx")

    def test_addition(self):
        # read cube
        c = CtxCube()
        c.read(self.cube000)
        d = c + 5
        self.assertEqual(c.cube[10][20][30] + 5, d.cube[10][20][30])


if __name__ == '__main__':
    unittest.main()
