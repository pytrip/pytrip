import imghdr
import unittest
import os
import tempfile
import glob
import logging

import shutil

from tests.test_base import get_files

import pytrip.utils.trip2dicom
import pytrip.utils.dicom2trip
import pytrip.utils.cubeslice

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


class TestDicom2Trip(unittest.TestCase):
    def test_check(self):
        self.assertRaises(SystemExit, pytrip.utils.dicom2trip.main, [])


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

    # def test_version(self):
    #     try:
    #         pytrip.utils.cubeslice.main(["--version"])
    #     except SystemExit as e:
    #         self.assertEqual(e.code, 0)

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
                                                     '-t', '6',
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
