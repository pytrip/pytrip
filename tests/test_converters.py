import unittest
import os
import tempfile

import argparse

from tests.test_base import get_files

import pytrip.utils.trip2dicom
import pytrip.utils.dicom2trip
import pytrip.utils.slicedos


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
        testdir = get_files()
        self.cube000 = os.path.join(testdir, "tst003000")

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

    def test_many_shield(self):
        ctx = os.path.join("tests", "res", "TST003", "tst003000.ctx.gz")
        dos = os.path.join("tests", "res", "TST003", "tst003001.dos.gz")
        pytrip.utils.slicedos.main([dos, ctx])


if __name__ == '__main__':
    unittest.main()
