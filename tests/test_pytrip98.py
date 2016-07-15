import unittest
import sys
import os

if sys.version_info >= (3,):
    import urllib.request as urllib2
else:
    import urllib2
import tarfile

import pytrip.utils.trip2dicom
import pytrip.utils.dicom2trip

import tempfile


class TestTrip2Dicom(unittest.TestCase):
    def setUp(self):
        # get plans from https://neptun.phys.au.dk/~bassler/TRiP/
        bname = "TST003"
        fname = bname + ".tar.gz"
        urllib2.urlretrieve("https://neptun.phys.au.dk/~bassler/TRiP/" + fname, fname)
        tfile = tarfile.open(fname, 'r:gz')
        tfile.extractall(".")

        self.cube000 = os.path.join(bname, "tst003000")

    def test_check(self):
        tmpdir = tempfile.mkdtemp()
        self.assertRaises(SystemExit, pytrip.utils.trip2dicom.main, [self.cube000, tmpdir])


class TestDicom2Trip(unittest.TestCase):
    def test_check(self):
        self.assertRaises(SystemExit, pytrip.utils.dicom2trip.main, [])


if __name__ == '__main__':
    unittest.main()
