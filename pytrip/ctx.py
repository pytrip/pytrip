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
The CTX module contains the CtxCube class which is inherited from the Cube class.
It is used for handling CT-data, both Voxelplan and Dicom.
"""
import os
import datetime
import copy
import logging

import numpy as np

from pytrip.error import InputError
from pytrip.cube import Cube

logger = logging.getLogger(__name__)


class CtxCube(Cube):
    """ Class for handling CT-data. In TRiP98 these are stored in VOXELPLAN format with the .ctx suffix.
    This class can also handle Dicom files.
    """
    data_file_extension = '.ctx'

    def __init__(self, cube=None):
        """ Creates an instance of a CtxCube.
        """
        super(CtxCube, self).__init__(cube)
        self.type = "CTX"

    def read_dicom(self, dcm):
        """ Imports CT-images from Dicom object.

        :param Dicom dcm: a Dicom object
        you can create Dicom object with pt.dicomhelper.read_dicom_dir(dicom_path)
        Don't confuse dicom object with filename or pydicom object
        """
        if "images" not in dcm:
            raise InputError("Data doesn't contain ct data")
        if not self.header_set:
            self._set_header_from_dicom(dcm)

        self.cube = np.zeros((self.dimz, self.dimy, self.dimx), dtype=np.int16)
        intersect = float(dcm["images"][0].RescaleIntercept)
        slope = float(dcm["images"][0].RescaleSlope)

        for i, dcm_image in enumerate(dcm["images"]):
            data = np.array(dcm_image.pixel_array) * slope + intersect
            self.cube[i][:][:] = data
        if len(self.slice_pos) > 1 and self.slice_pos[1] < self.slice_pos[0]:
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
        ds.SamplesPerPixel = 1
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
        ds.Manufacturer = self.creation_info  # Manufacturer tag, 0x0008,0x0070 (type LO - Long String)

        ds.KVP = ''  # KVP tag 0x0018, 0x0060

        ds.AcquisitionNumber = '1'  # AcquisitionNumber tag 0x0020, 0x0012 (type IS - Integer String)

        import uuid
        for i, cube in enumerate(self.cube):
            _ds = copy.deepcopy(ds)
            _ds.ImagePositionPatient = ["{:.3f}".format(self.xoffset),
                                        "{:.3f}".format(self.yoffset),
                                        "{:.3f}".format(self.slice_pos[i])]

            if ds.SOPInstanceUID.startswith('2.25.'):
                # UUID based UIDs
                # example: 2.25.137355362850316362338405159557803441718
                uuid_part_str = ds.SOPInstanceUID[len('2.25.'):len('2.25.') + 32]  # extract 32bit fragment of
                # last part of UID, as string

                uuid_object = uuid.UUID(int=int(uuid_part_str))   # convert to UID object, to be able to manipulate it
                uuid_list = list(uuid_object.fields)              # get list of fields, to be able to edit it
                uuid_list[-1] = i + 1  # replace clock_seq part of uuid with sequential number
                current_uuid = uuid.UUID(fields=uuid_list)        # create uuid object back from updated list
                current_sop_uid = '2.25.{0}'.format(current_uuid.int)  # create back an UID
            else:
                # ISO based UIDS
                # example: 1.2.826.0.1.3680043.8.498.255851143265846913128620976
                sop_uid_list = ds.SOPInstanceUID.split('.')
                current_sop_uid = '.'.join(sop_uid_list[:-1] + [str(i + 1)])  # replace last part of UID with a number

            _ds.SOPInstanceUID = current_sop_uid
            _ds.SliceLocation = str(self.slice_pos[i])
            _ds.InstanceNumber = str(i + 1)
            pixel_array = np.zeros((_ds.Rows, _ds.Columns), dtype=self.pydata_type)
            pixel_array[:][:] = cube[:][:]
            _ds.PixelData = pixel_array.tostring()
            data.append(_ds)
        return data

    def write_dicom(self, directory):
        """ Write CT-data to disk, in Dicom format.

        :param str directory: directory to write to. If directory does not exist, it will be created.
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        dcm_list = self.create_dicom()
        for dcm_item in dcm_list:
            dcm_item.save_as(os.path.join(directory, "CT.PYTRIP.{:d}.dcm".format(dcm_item.InstanceNumber)))
