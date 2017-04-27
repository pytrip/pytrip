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
import imghdr
import unittest
import os
import tempfile
import logging
import shutil

import pytrip.utils.rst_plot
import pytrip.utils.dvhplot

from tests.base import get_files

logger = logging.getLogger(__name__)


class TestRstPlot(unittest.TestCase):
    def setUp(self):
        testdir = get_files()
        self.rst_file = os.path.join(testdir, "tst003001.rst")

    def test_check(self):
        self.assertRaises(SystemExit, pytrip.utils.rst_plot.main, [])

    def test_version(self):
        try:
            pytrip.utils.rst_plot.main(["--version"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    def test_generate(self):
        fd, outfile = tempfile.mkstemp(suffix='.png')

        # convert CT cube to DICOM
        pytrip.utils.rst_plot.main([self.rst_file, outfile])

        # check if destination file is not empty
        self.assertTrue(os.path.exists(outfile))
        self.assertGreater(os.path.getsize(outfile), 0)
        self.assertEqual(imghdr.what(outfile), 'png')

        os.close(fd)  # Windows needs it
        os.remove(outfile)  # we need only temp filename, not the file


class TestDvhPlot(unittest.TestCase):
    def setUp(self):
        self.dir_path = os.path.join("tests", "res", "TST003")

        self.vdx = os.path.join(self.dir_path, "tst003000.vdx")
        logger.info("Loading vdx file " + self.vdx)

        self.dos = os.path.join(self.dir_path, "tst003001.dos.gz")
        logger.info("Loading dos file " + self.dos)

        self.let = os.path.join(self.dir_path, "tst003001.dosemlet.dos.gz")
        logger.info("Loading let file " + self.let)

    def test_help(self):
        try:
            pytrip.utils.dvhplot.main(["--help"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    def test_version(self):
        try:
            pytrip.utils.dvhplot.main(["--version"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    def test_noarg(self):
        try:
            pytrip.utils.dvhplot.main([])
        except SystemExit as e:
            self.assertEqual(e.code, 2)

    def test_relative_dos_plot(self):
        working_dir = tempfile.mkdtemp()  # make temp working dir for output file
        output_file = os.path.join(working_dir, "foo.png")

        pytrip.utils.dvhplot.main(args=[self.dos, self.vdx, 'target', '-l', '-v', '-o', output_file])

        logger.info("Checking if " + output_file + " is PNG")
        self.assertEqual(imghdr.what(output_file), 'png')

        logger.info("Removing " + working_dir)
        shutil.rmtree(working_dir)

    def test_absolute_dos_plot(self):
        working_dir = tempfile.mkdtemp()  # make temp working dir for output file
        output_file = os.path.join(working_dir, "foo.png")

        pytrip.utils.dvhplot.main(args=[self.dos, self.vdx, 'target', '-l', '-v', '-d 2.0', '-o', output_file])

        logger.info("Checking if " + output_file + " is PNG")
        self.assertEqual(imghdr.what(output_file), 'png')

        logger.info("Removing " + working_dir)
        shutil.rmtree(working_dir)

    def test_let_plot(self):
        working_dir = tempfile.mkdtemp()  # make temp working dir for output file
        output_file = os.path.join(working_dir, "foo.png")

        pytrip.utils.dvhplot.main(args=[self.let, self.vdx, 'target', '-l', '-v', '-o', output_file])

        logger.info("Checking if " + output_file + " is PNG")
        self.assertEqual(imghdr.what(output_file), 'png')

        logger.info("Removing " + working_dir)
        shutil.rmtree(working_dir)


if __name__ == '__main__':
    unittest.main()
