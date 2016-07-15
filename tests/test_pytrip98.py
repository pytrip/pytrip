import unittest
import pytrip.utils.trip2dicom
import pytrip.utils.dicom2trip


class TestTrip2Dicom(unittest.TestCase):
    def test_check(self):
        self.assertRaises(SystemExit, pytrip.utils.trip2dicom.main, [])


class TestDicom2Trip(unittest.TestCase):
    def test_check(self):
        self.assertRaises(SystemExit, pytrip.utils.dicom2trip.main, [])


if __name__ == '__main__':
    unittest.main()
