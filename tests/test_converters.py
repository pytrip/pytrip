import imghdr
import unittest
import os
import tempfile
import glob
import logging

from tests.test_base import get_files

import pytrip.utils.trip2dicom
import pytrip.utils.dicom2trip
import pytrip.utils.slicedos

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


class TestDosSlicer(unittest.TestCase):
    def setUp(self):
        self.dir_path = os.path.join("tests", "res", "TST003")

        self.ctx = os.path.join(self.dir_path, "tst003000.ctx.gz")
        logger.info("Loading ctx file " + self.ctx)

        self.dos = os.path.join(self.dir_path, "tst003001.dos.gz")
        logger.info("Loading dos file " + self.dos)

    def test_help(self):
        try:
            pytrip.utils.slicedos.main(["--help"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    # def test_version(self):
    #     try:
    #         pytrip.utils.slicedos.main(["--version"])
    #     except SystemExit as e:
    #         self.assertEqual(e.code, 0)

    def test_noarg(self):
        try:
            pytrip.utils.slicedos.main([])
        except SystemExit as e:
            self.assertEqual(e.code, 2)

    def test_convert_all(self):
        pytrip.utils.slicedos.main(args=[self.dos, self.ctx])
        output_file_list = glob.glob(os.path.join(self.dir_path, "*.png"))

        logger.info("Checking if number of output files is sufficient")
        self.assertEqual(len(output_file_list), 300)

        for output_file in output_file_list:
            logger.info("Checking if " + output_file + " is PNG")
            self.assertEqual(imghdr.what(output_file), 'png')

    # TODO should be activated once it is possible to save output files to some directory
    # def test_convert_one(self):
    #     pytrip.utils.slicedos.main(args=[self.dos, self.ctx, "-f", '5' , "-t", '6'])
    #     output_file_list = glob.glob(os.path.join(self.dir_path, "*.png"))
    #
    #     logger.info("Checking if number of output files is sufficient")
    #     self.assertEqual(len(output_file_list), 1)
    #
    #     for output_file in output_file_list:
    #         logger.info("Checking if " + output_file + " is PNG")
    #         self.assertEqual(imghdr.what(output_file), 'png')


if __name__ == '__main__':
    unittest.main()
