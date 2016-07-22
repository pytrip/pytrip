import unittest
import os
import tempfile

from tests.test_base import get_files

import pytrip.utils.trip2dicom
import pytrip.utils.dicom2trip


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


if __name__ == '__main__':
    unittest.main()
