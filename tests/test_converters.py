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
import sys
import tempfile
import glob
import logging
import shutil

import pytrip.utils.trip2dicom
import pytrip.utils.dicom2trip
import pytrip.utils.cubeslice
import pytrip.utils.rst2sobp
import pytrip.utils.gd2dat
import pytrip.utils.gd2agr
import pytrip.utils.bevlet2oer

from tests.base import get_files

logger = logging.getLogger(__name__)


class TestTrip2Dicom(unittest.TestCase):
    def setUp(self):
        testdir = get_files()
        self.cube000 = os.path.join(testdir, "tst003000")

    def test_generate(self):
        # create temp dir
        tmpdir = tempfile.mkdtemp()

        # convert CT cube to DICOM
        pytrip.utils.trip2dicom.main([self.cube000, tmpdir])

        # check if destination directory is not empty
        self.assertTrue(os.listdir(tmpdir))

    def test_version(self):
        try:
            pytrip.utils.trip2dicom.main(["--version"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)


class TestRst2SOBP(unittest.TestCase):
    def setUp(self):
        testdir = get_files()
        self.rst_file = os.path.join(testdir, "tst003001.rst")

    def test_generate(self):
        fd, outfile = tempfile.mkstemp()

        # convert CT cube to DICOM
        pytrip.utils.rst2sobp.main([self.rst_file, outfile])

        # check if destination file is not empty
        self.assertTrue(os.path.exists(outfile))
        self.assertGreater(os.path.getsize(outfile), 0)

        os.close(fd)  # Windows needs it
        os.remove(outfile)  # we need only temp filename, not the file

    def test_version(self):
        try:
            pytrip.utils.rst2sobp.main(["--version"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)


class TestGd2Dat(unittest.TestCase):
    def setUp(self):
        testdir = get_files()
        self.gd_file = os.path.join(testdir, "tst003001.bev.gd")

    def test_generate(self):
        fd, outfile = tempfile.mkstemp()

        # convert CT cube to DICOM
        pytrip.utils.gd2dat.main([self.gd_file, outfile])

        # check if destination file is not empty
        self.assertTrue(os.path.exists(outfile))
        self.assertGreater(os.path.getsize(outfile), 0)

        os.close(fd)  # Windows needs it
        os.remove(outfile)  # we need only temp filename, not the file

    def test_version(self):
        try:
            pytrip.utils.gd2dat.main(["--version"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)


class TestGd2Agr(unittest.TestCase):
    def setUp(self):
        testdir = get_files()
        self.gd_file = os.path.join(testdir, "tst003001.bev.gd")

    def test_generate(self):
        fd, outfile = tempfile.mkstemp()

        # convert CT cube to DICOM
        pytrip.utils.gd2agr.main([self.gd_file, outfile])

        # check if destination file is not empty
        self.assertTrue(os.path.exists(outfile))
        self.assertGreater(os.path.getsize(outfile), 0)

        os.close(fd)  # Windows needs it
        os.remove(outfile)  # we need only temp filename, not the file

    def test_version(self):
        try:
            pytrip.utils.gd2agr.main(["--version"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)


class TestBevLet2Oer(unittest.TestCase):
    def setUp(self):
        testdir = get_files()
        self.gd_file = os.path.join(testdir, "tst003001.bevlet.gd")

    def test_check(self):
        self.assertRaises(SystemExit, pytrip.utils.bevlet2oer.main, [])

    def test_version(self):
        try:
            pytrip.utils.bevlet2oer.main(["--version"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    def test_generate(self):
        fd, outfile = tempfile.mkstemp()

        # convert CT cube to DICOM
        pytrip.utils.bevlet2oer.main([self.gd_file, outfile])

        # check if destination file is not empty
        self.assertTrue(os.path.exists(outfile))
        self.assertGreater(os.path.getsize(outfile), 0)

        os.close(fd)  # Windows needs it
        os.remove(outfile)  # we need only temp filename, not the file


class TestDicom2Trip(unittest.TestCase):
    def test_check(self):
        self.assertRaises(SystemExit, pytrip.utils.dicom2trip.main, [])

    def test_version(self):
        try:
            pytrip.utils.dicom2trip.main(["--version"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)


class TestSpc2Pdf(unittest.TestCase):
    def test_check(self):
        if sys.version_info[0] == 3 and sys.version_info[1] == 2:
            retcode = pytrip.utils.spc2pdf.main()
            self.assertEqual(retcode, 1)
        else:
            # Some import hacking needed by Appveyor
            from pytrip.utils import spc2pdf  # noqa F401
            self.assertRaises(SystemExit, pytrip.utils.spc2pdf.main, [])

    def test_version(self):
        try:
            # Some import hacking needed by Appveyor
            from pytrip.utils import spc2pdf  # noqa F401
            pytrip.utils.spc2pdf.main(["--version"])
        except SystemExit as e:
            if sys.version_info[0] == 3 and sys.version_info[1] == 2:
                self.assertEqual(e.code, 1)
            else:
                self.assertEqual(e.code, 0)


class TestCubeSlicer(unittest.TestCase):
    def setUp(self):
        self.dir_path = os.path.join("tests", "res", "TST003")

        self.ctx = os.path.join(self.dir_path, "tst003000.ctx.gz")
        logger.info("Loading ctx file " + self.ctx)

        self.dos = os.path.join(self.dir_path, "tst003001.dos.gz")
        logger.info("Loading dos file " + self.dos)

        self.let = os.path.join(self.dir_path, "tst003001.dosemlet.dos.gz")
        logger.info("Loading dos file " + self.dos)

    def test_help(self):
        try:
            pytrip.utils.cubeslice.main(["--help"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    def test_version(self):
        try:
            pytrip.utils.cubeslice.main(["--version"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    def test_noarg(self):
        try:
            pytrip.utils.cubeslice.main([])
        except SystemExit as e:
            self.assertEqual(e.code, 2)

    def test_convert_all(self):
        working_dir = tempfile.mkdtemp()  # make temp working dir for converter output files

        pytrip.utils.cubeslice.main(args=['--data', self.dos, '--ct', self.ctx, '-o', working_dir])
        output_file_list = glob.glob(os.path.join(working_dir, "*.png"))

        logger.info("Checking if number of output files is sufficient")
        self.assertEqual(len(output_file_list), 300)

        for output_file in output_file_list:
            logger.info("Checking if " + output_file + " is PNG")
            self.assertEqual(imghdr.what(output_file), 'png')

        logger.info("Removing " + working_dir)
        shutil.rmtree(working_dir)

    def test_convert_one(self):
        working_dir = tempfile.mkdtemp()  # make temp working dir for converter output files

        ret_code = pytrip.utils.cubeslice.main(args=['--data', self.dos,
                                                     '--ct', self.ctx,
                                                     '-f', '5',
                                                     '-t', '5',
                                                     '-o', working_dir])
        self.assertEqual(ret_code, 0)

        output_file_list = glob.glob(os.path.join(working_dir, "*.png"))

        logger.info("Checking if number of output files is sufficient")
        self.assertEqual(len(output_file_list), 1)

        for output_file in output_file_list:
            logger.info("Checking if " + output_file + " is PNG")
            self.assertEqual(imghdr.what(output_file), 'png')

        logger.info("Removing " + working_dir)
        shutil.rmtree(working_dir)


if __name__ == '__main__':
    unittest.main()
