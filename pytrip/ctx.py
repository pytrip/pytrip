"""
    This file is part of PyTRiP.

    libdedx is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    libdedx is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with libdedx.  If not, see <http://www.gnu.org/licenses/>
"""
import numpy
from pytrip.header import *
from pytrip.error import *
from pytrip.cube import *

try:
    from dicom.dataset import Dataset, FileDataset
    import dicom

    _dicom_loaded = True
except:
    _dicom_loaded = False

__author__ = "Niels Bassler and Jakob Toftegaard"
__version__ = "1.0"
__email__ = "bassler@phys.au.dk"


class CtxCube(Cube):
    def __init__(self, cube=None):
        super(CtxCube, self).__init__(cube)
        self.type = "CTX"

    def read_dicom(self, dcm):
        if not dcm.has_key("images"):
            raise InputError("Data doesn't contain ct data")
        if self.header_set is False:
            self.read_dicom_header(dcm)

        self.cube = np.zeros((self.dimz, self.dimy, self.dimx), dtype=numpy.int16)
        intersect = float(dcm["images"][0].RescaleIntercept)
        slope = float(dcm["images"][0].RescaleSlope)

        for i in range(len(dcm["images"])):
            data = numpy.array(dcm["images"][i].pixel_array) * slope + intersect
            self.cube[i][:][:] = data
        if self.slice_pos[1] < self.slice_pos[0]:
            self.slice_pos.reverse()
            self.zoffset = self.slice_pos[0]
            self.cube = self.cube[::-1]

    def create_dicom(self):
        data = []
        if _dicom_loaded is False:
            raise ModuleNotLoadedError("Dicom")
        if self.header_set is False:
            raise InputError("Header not loaded")

        for i in range(len(self.cube)):
            ds = self.create_dicom_base()
            ds.Modality = 'CT'
            ds.SamplesperPixel = 1
            ds.BitsAllocated = self.num_bytes * 8
            ds.BitsStored = self.num_bytes * 8
            ds.HighBit = self.num_bytes * 8 - 1
            ds.PatientPosition = 'HFS'
            ds.RescaleIntercept = 0.0
            ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']

            ds.PatientPosition = 'HFS'
            ds.SeriesInstanceUID = '2.16.840.1.113662.2.12.0.3057.1241703565.43'  # !!!!!!!!!!
            ds.RescaleSlope = 1.0
            ds.PixelRepresentation = 1
            ds.ImagePositionPatient = ["%.3f" % (self.xoffset * self.pixel_size),
                                       "%.3f" % (self.yoffset * self.pixel_size), "%.3f" % (self.slice_pos[i])]
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            ds.SOPInstanceUID = '2.16.1.113662.2.12.0.3057.1241703565.' + str(i + 1)

            ds.SeriesDate = '19010101'  # !!!!!!!!
            ds.ContentDate = '19010101'  # !!!!!!
            ds.SeriesTime = '000000'  # !!!!!!!!!
            ds.ContentTime = '000000'  # !!!!!!!!!

            ds.SliceLocation = str(self.slice_pos[i])
            ds.InstanceNumber = str(i + 1)
            pixel_array = numpy.zeros((ds.Rows, ds.Columns), dtype=self.pydata_type)
            pixel_array[:][:] = self.cube[i][:][:]
            ds.PixelData = pixel_array.tostring()
            ds.pixel_array = pixel_array
            data.append(ds)
        return data

    def write(self, path):
        f_split = os.path.splitext(path)
        header_file = f_split[0] + ".hed"
        ctx_file = f_split[0] + ".ctx"
        self.write_trip_header(header_file)
        self.write_trip_data(ctx_file)

    def write_dicom(self, path):
        dcm_list = self.create_dicom()
        for i in range(len(dcm_list)):
            dcm_list[i].save_as(os.path.join(path, "ct.%d.dcm" % (dcm_list[i].InstanceNumber - 1)))
