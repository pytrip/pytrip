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

from pytrip.ctx import CtxCube
import tests.test_base


class TestCtx(unittest.TestCase):
    def setUp(self):
        testdir = tests.test_base.get_files()
        self.cube000 = os.path.join(testdir, "tst003000.ctx")

    def test_read(self):
        c = CtxCube()
        c.read(self.cube000)

    def test_write(self):
        # read original cube and calculate hashsum
        c = CtxCube()
        c.read(self.cube000)
        original_md5 = hashlib.md5(gzip.open(self.cube000 + ".gz", 'rb').read()).hexdigest()

        # save cube and calculate hashsum
        fd, outfile = tempfile.mkstemp()
        c.write(outfile)
        generated_md5 = hashlib.md5(open(outfile + ".ctx", 'rb').read()).hexdigest()

        # compare checksums
        self.assertEqual(original_md5, generated_md5)

        # TODO compare also header files

    def test_addition(self):
        # read cube
        c = CtxCube()
        c.read(self.cube000)
        d = c + 5
        self.assertEqual(c.cube[10][20][30] + 5, d.cube[10][20][30])


if __name__ == '__main__':
    unittest.main()
