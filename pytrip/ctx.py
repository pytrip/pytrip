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
import logging

import numpy as np

from pytrip.dicom.ct import create_dicom
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

        for i in range(len(dcm["images"])):
            intersect = float(dcm["images"][i].RescaleIntercept)
            slope = float(dcm["images"][i].RescaleSlope)
            data = np.array(dcm["images"][i].pixel_array) * slope + intersect
            self.cube[i][:][:] = data
        if len(self.slice_pos) > 1 and self.slice_pos[1] < self.slice_pos[0]:
            self.slice_pos.reverse()
            self.zoffset = self.slice_pos[0]
            self.cube = self.cube[::-1]

    def write_dicom(self, directory):
        """ Write CT-data to disk, in Dicom format.

        :param str directory: directory to write to. If directory does not exist, it will be created.
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        dcm_list = create_dicom(self)
        for dcm_item in dcm_list:
            output_filename = "CT.PYTRIP.{:d}.dcm".format(dcm_item.InstanceNumber)
            logger.info("Saving {}".format(output_filename))
            dcm_item.save_as(os.path.join(directory, output_filename))
