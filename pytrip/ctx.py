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
import datetime
import copy

import numpy as np

from pytrip.error import InputError
from pytrip.cube import Cube


class CtxCube(Cube):
    """ Class for handling CT-data. In TRiP98 these are stored in VOXELPLAN format with the .ctx suffix.
    This class can also handle Dicom files.
    """
    data_file_extension = ".ctx"
    header_file_extension = ".hed"

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
        # ds.SeriesInstanceUID is created in the top-level cube class
        # ds.SeriesInstanceUID = '2.16.840.1.113662.2.12.0.3057.1241703565.43'
        ds.RescaleSlope = 1.0
        ds.PixelRepresentation = 1
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage SOP Class

        # .HED files do not carry any time stamp (other than the usual file meta data)
        # so let's just fill it with current times. (Can be overridden by user)
        ds.SeriesDate = datetime.datetime.today().strftime('%Y%m%d')
        ds.ContentDate = datetime.datetime.today().strftime('%Y%m%d')
        ds.SeriesTime = datetime.datetime.today().strftime('%H%M%S')
        ds.ContentTime = datetime.datetime.today().strftime('%H%M%S')

        # Eclipse tags

        # Manufacturer of the equipment that produced the composite instances.
        ds.Manufacturer = self.creation_info  # Manufacturer tag,0x0008, 0x0070

        ds.KVP = ''  # KVP tag 0x0018, 0x0060

        ds.AcquisitionNumber = '1'  # AcquisitionNumber tag 0x0020, 0x0012 (type IS - Integer String)

        for i in range(len(self.cube)):
            _ds = copy.deepcopy(ds)
            _ds.ImagePositionPatient = ["%.3f" % (self.xoffset * self.pixel_size),
                                        "%.3f" % (self.yoffset * self.pixel_size),
                                        "%.3f" % (self.slice_pos[i])]

            _ds.SOPInstanceUID = '2.16.1.113662.2.12.0.3057.1241703565.' + str(i + 1)
            _ds.SliceLocation = str(self.slice_pos[i])
            _ds.InstanceNumber = str(i + 1)
            pixel_array = np.zeros((_ds.Rows, _ds.Columns), dtype=self.pydata_type)
            pixel_array[:][:] = self.cube[i][:][:]
            _ds.PixelData = pixel_array.tostring()
            _ds.pixel_array = pixel_array
            data.append(_ds)
        return data

    def write(self, path):
        """ Write CT-data to disk, in TRiP98/Voxelplan format.

        This method will build and write both the .hed and .ctx file.

        :param str path: path to header file, data file or basename (without extension)
        """

        header_file, ctx_file = self.parse_path(path_name=path)
        self.write_trip_header(header_file)
        self.write_trip_data(ctx_file)

    def write_dicom(self, directory):
        """ Write CT-data to disk, in Dicom format.

        :param str directory: directory to write to. If directory does not exist, it will be created.
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        dcm_list = self.create_dicom()
        for dcm_item in dcm_list:
            dcm_item.save_as(os.path.join(directory, "CT.PYTRIP.{:d}.dcm".format(dcm_item.InstanceNumber)))
