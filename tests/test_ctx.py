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
import gzip
import hashlib
import os
import tempfile
import unittest
import logging

from pytrip import DosCube
from pytrip.ctx import CtxCube
import tests.test_base

logger = logging.getLogger(__name__)


class TestCtx(unittest.TestCase):
    def setUp(self):
        testdir = tests.test_base.get_files()
        self.cube000 = os.path.join(testdir, "tst003000")

    def test_read(self):
        c = CtxCube()
        c.read(self.cube000)

    def read_and_write_cube(self, path):

        logger.info("Testing cube from path " + path)

        # read original cube and calculate hashsum
        c = CtxCube()
        c.read(path)

        _, data_file = CtxCube.parse_path(self.cube000)
        data_file_path = CtxCube.discover_file(data_file)

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
        c.write(outfile)   # this will write outfile+".ctx"  and outfile+".hed"
        f = open(outfile + ".ctx", 'rb')
        generated_md5 = hashlib.md5(f.read()).hexdigest()
        f.close()
        logger.debug("Removing " + outfile + ".ctx")
        os.remove(outfile + ".ctx")
        logger.debug("Removing " + outfile + ".hed")
        os.remove(outfile + ".hed")
        # compare checksums
        self.assertEqual(original_md5, generated_md5)

    def test_write(self):
        possible_names = [self.cube000,
                          self.cube000 + ".ctx", self.cube000 + ".hed",
                          self.cube000 + ".hed.gz", self.cube000 + ".ctx.gz"]

        for name in possible_names:
            self.read_and_write_cube(name)

    def test_addition(self):
        # read cube
        c = CtxCube()
        c.read(self.cube000)
        d = c + 5
        self.assertEqual(c.cube[10][20][30] + 5, d.cube[10][20][30])

    def test_filename_parsing(self):
        bare_name = "frodo_baggins"
        cube_ext = ".ctx"
        header_ext = ".hed"
        gzip_ext = ".gz"

        # all possible reasonable names for input file
        testing_names = (bare_name,
                         bare_name + cube_ext, bare_name + cube_ext + gzip_ext,
                         bare_name + header_ext, bare_name + header_ext + gzip_ext)

        # loop over all names
        for name in testing_names:
            logger.info("Parsing " + name)
            header_filename, cube_filename = CtxCube.parse_path(name)

            self.assertIsNotNone(header_filename)
            self.assertIsNotNone(cube_filename)

            logger.info("Parsing output: " + header_filename + " , " + cube_filename)

            # test if got what was expected
            self.assertEqual(header_filename, bare_name + header_ext)
            self.assertEqual(cube_filename, bare_name + cube_ext)

        dos_cube_ext = ".dos"

        # all possible reasonable names for input file
        testing_names = (bare_name,
                         bare_name + dos_cube_ext, bare_name + dos_cube_ext + gzip_ext,
                         bare_name + header_ext, bare_name + header_ext + gzip_ext)

        # loop over all names
        for name in testing_names:
            logger.info("Parsing " + name)
            dos_header_filename, dos_cube_filename = DosCube.parse_path(name)

            self.assertIsNotNone(dos_header_filename)
            self.assertIsNotNone(dos_cube_filename)

            logger.info("Parsing output: " + dos_header_filename + " , " + dos_cube_filename)

            # test if got what was expected
            self.assertEqual(dos_header_filename, bare_name + header_ext)
            self.assertEqual(dos_cube_filename, bare_name + dos_cube_ext)


if __name__ == '__main__':
    unittest.main()
