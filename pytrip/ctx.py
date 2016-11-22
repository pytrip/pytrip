#
#    Copyright (C) 2010-2016 PyTRiP98 Developers.
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
The CTX module contains the CtxCube class which is inherited from the Cube class.
It is used for handling CT-data, both Voxelplan and Dicom.
"""
import os

import numpy as np

from pytrip.error import InputError
from pytrip.cube import Cube


class CtxCube(Cube):
    """ Class for handling CT-data. In TRiP98 these are stored in VOXELPLAN format with the .ctx suffix.
    This class can also handle Dicom files.
    """
    data_file_extension = "ctx"

    def __init__(self, cube=None):
        super(CtxCube, self).__init__(cube)
        self.type = "CTX"

    def read_dicom(self, dcm):
        """ Imports CT-images from Dicom object.

        :param Dicom dcm: a Dicom object
        """
        if "images" not in dcm:
            raise InputError("Data doesn't contain ct data")
        if not self.header_set:
            self.read_dicom_header(dcm)

        self.cube = np.zeros((self.dimz, self.dimy, self.dimx), dtype=np.int16)
        intersect = float(dcm["images"][0].RescaleIntercept)
        slope = float(dcm["images"][0].RescaleSlope)

        for i in range(len(dcm["images"])):
            data = np.array(dcm["images"][i].pixel_array) * slope + intersect
            self.cube[i][:][:] = data
        if self.slice_pos[1] < self.slice_pos[0]:
            self.slice_pos.reverse()
            self.zoffset = self.slice_pos[0]
            self.cube = self.cube[::-1]

    def create_dicom(self):
        """ Creates a Dicom object from self.

        This function can be used to convert a TRiP98 CTX file to Dicom format.

        :returns: A Dicom object.
        """
        data = []

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
            ds.SeriesInstanceUID = '2.16.840.1.113662.2.12.0.3057.1241703565.43'
            ds.RescaleSlope = 1.0
            ds.PixelRepresentation = 1
            ds.ImagePositionPatient = ["%.3f" % (self.xoffset * self.pixel_size),
                                       "%.3f" % (self.yoffset * self.pixel_size),
                                       "%.3f" % (self.slice_pos[i])]
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage SOP Class
            ds.SOPInstanceUID = '2.16.1.113662.2.12.0.3057.1241703565.' + str(i + 1)

            ds.SeriesDate = '19010101'  # !!!!!!!!
            ds.ContentDate = '19010101'  # !!!!!!
            ds.SeriesTime = '000000'  # !!!!!!!!!
            ds.ContentTime = '000000'  # !!!!!!!!!

            ds.SliceLocation = str(self.slice_pos[i])
            ds.InstanceNumber = str(i + 1)
            pixel_array = np.zeros((ds.Rows, ds.Columns), dtype=self.pydata_type)
            pixel_array[:][:] = self.cube[i][:][:]
            ds.PixelData = pixel_array.tostring()
            ds.pixel_array = pixel_array
            data.append(ds)
        return data

    def write(self, path):
        """ Write CT-data to disk, in TRiP98/Voxelplan format.

        This method will build and write both the .hed and .ctx file.

        :param str path: Path, any file extentions will be ignored.
        """
        f_split = os.path.splitext(path)
        header_file = f_split[0] + ".hed"
        ctx_file = f_split[0] + ".ctx"
        self.write_trip_header(header_file)
        self.write_trip_data(ctx_file)

    def write_dicom(self, path):
        """ Write CT-data to disk, in Dicom format.

        :param str path: Full path,  including file extention.
        """
        dcm_list = self.create_dicom()
        for i in range(len(dcm_list)):
            dcm_list[i].save_as(os.path.join(path, "ct.%d.dcm" % (dcm_list[i].InstanceNumber - 1)))
