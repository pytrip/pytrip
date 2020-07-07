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
This module provides the DosCube class, which the user should use for handling plan-generated dose distributions.
"""
import datetime
import logging
import os
import warnings

import numpy as np

from pydicom import uid
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence

from pytrip.cube import Cube
from pytrip.error import InputError


class DosCube(Cube):
    """ Class for handling Dose data. In TRiP98 these are stored in VOXELPLAN format with the .dos/.DOS suffix.
    This class can also handle DICOM files.
    """
    data_file_extension = '.dos'
    allowed_suffix = ('phys', 'bio', 'rbe', 'svv', 'alpha', 'beta')

    def __init__(self, cube=None):
        """ Creates a DosCube instance.
        If cube is provided, then UIDs are inherited from cube.
        """
        super(DosCube, self).__init__(cube)
        self.type = "DOS"
        self.target_dose = 0.0  # Target dose in Gy or Gy(RBE)

        # UIDs unique for whole structure set
        # generation of UID is done here in init, the reason why we are not generating them in create_dicom
        # method is that subsequent calls to write method shouldn't changed UIDs
        if not cube:
            self._dicom_study_instance_uid = uid.generate_uid(prefix=None)

        self.dicom_data = getattr(cube, "dicom_data", {})
        self._plan_dicom_series_instance_uid = uid.generate_uid(prefix=None)
        self._dose_dicom_series_instance_uid = uid.generate_uid(prefix=None)
        self._dose_dicom_SOP_instance_uid = uid.generate_uid(prefix=None)

    def read_dicom(self, dcm):
        """ Imports the dose distribution from DICOM object.

        :param DICOM dcm: a DICOM object
        """
        if "rtdose" not in dcm:
            raise InputError("Data doesn't contain dose information")
        if self.header_set is False:
            self._set_header_from_dicom(dcm)
        self.cube = np.zeros((self.dimz, self.dimy, self.dimx))
        for i, item in enumerate(dcm["rtdose"].pixel_array):
            self.cube[i][:][:] = item

    def calculate_dvh(self, voi):
        """
        Calculate DHV for given VOI. Dose is given in relative units (target dose = 1.0).
        In case VOI lies outside the cube, then None is returned.

        :param voi: VOI for which DHV should be calculated
        :return: (dvh, min_dose, max_dose, mean, area) tuple. dvh - 2D array holding DHV histogram,
        min_dose and max_dose, mean_dose - obvious, mean_volume - effective volume dose.
        """

        warnings.warn(
            "The function calculate_dvh is deprecated, and is replaced with the pytrip.VolHist object.",
            DeprecationWarning
        )
        from pytrip import pytriplib
        z_pos = 0  # z position
        voxel_size = np.array([self.pixel_size, self.pixel_size, self.slice_distance])
        # in TRiP98 dose is stored in relative numbers, target dose is set to 1000 (and stored as 2-bytes ints)
        maximum_dose = 1500  # do not change, same value is hardcoded in filter_point.c (calculate_dvh_slice method)
        dose_bins = np.zeros(maximum_dose)  # placeholder for DVH, filled with zeros
        voi_and_cube_intersect = False
        for i in range(self.dimz):
            z_pos += self.slice_distance
            slice = voi.get_slice_at_pos(z_pos)
            if slice is not None:   # VOI intersects with this slice
                voi_and_cube_intersect = True
                dose_bins += pytriplib.calculate_dvh_slice(self.cube[i],
                                                           np.array(slice.contours[0].contour),
                                                           voxel_size)

        if voi_and_cube_intersect:
            sum_of_doses = sum(dose_bins)
            # np.cumsum - cumulative sum of array along the axis
            # we calculate is backwards and revert to get a plot which is monotonically decreasing
            # normalization is needed to get maximum values on Y axis to be <= 1
            dvh_x = np.arange(start=0.0, stop=1500.0)
            dvh_y = np.cumsum(dose_bins[::-1])[::-1] / sum_of_doses

            min_dose = np.where(dvh_y >= 0.98)[0][-1]
            max_dose = np.where(dvh_y <= 0.02)[0][0]

            # mean = \sum_i=1^1500 f_i * d_i , where:
            #   f_i = dose_bin(i) / \sum_i dose_bin(i)  - frequency of dose at index i
            #   d_i = dvx_i(i) = i                      - dose at index i (equal to i)
            #             dose goes from 0 to 1500 and is integer
            mean_dose = np.dot(dose_bins, dvh_x) / sum_of_doses

            # if full voi is irradiated with target dose, then it should be equal to VOI volume
            mean_volume = sum_of_doses * voxel_size[0] * voxel_size[1] * voxel_size[2]

            # TRiP98 target dose is 1000, we renormalize to 1.0
            min_dose /= 1000.0
            max_dose /= 1000.0
            mean_dose /= 1000.0
            mean_volume /= 1000.0
            dvh_x /= 1000.0

            dvh = np.column_stack((dvh_x, dvh_y))

            return dvh, min_dose, max_dose, mean_dose, mean_volume
        return None

    def write_dvh(self, voi, filename):
        """
        Save DHV for given VOI to the file.
        """
        warnings.warn(
            "The method write_dvh() is deprecated, and is replaced with the pytrip.VolHist object.",
            DeprecationWarning
        )
        dvh_tuple = self.calculate_dvh(voi)
        if dvh_tuple is None:
            logging.warning("Voi {:s} outside the cube".format(voi.get_name()))
        else:
            dvh, min_dose, max_dose, mean_dose, mean_volume = dvh_tuple
            np.savetxt(dvh, filename)

    def create_dicom_plan(self):
        """ Create a dummy DICOM RT-plan object.

        The only data which is forwarded to this object, is self.patient_name.
        :returns: a DICOM RT-plan object.
        """
        meta = Dataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        meta.MediaStorageSOPInstanceUID = "1.2.3"
        meta.ImplementationClassUID = "1.2.3.4"
        meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian  # Implicit VR Little Endian - Default Transfer Syntax

        ds = FileDataset("file", {}, file_meta=meta, preamble=b"\0" * 128)
        if self.cube is not None:
            ds.PatientName = self.patient_name
            ds.Manufacturer = self.creation_info  # Manufacturer tag, 0x0008,0x0070 (type LO - Long String)
        else:
            ds.PatientName = ''
            ds.Manufacturer = ''  # Manufacturer tag, 0x0008,0x0070 (type LO - Long String)

        ds.PatientsName = self.patient_name

        if self.patient_id in (None, ''):
            ds.PatientID = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
        else:
            ds.PatientID = self.patient_id  # Patient ID tag 0x0010,0x0020 (type LO - Long String)
        ds.PatientsSex = ''  # Patient's Sex tag 0x0010,0x0040 (type CS - Code String)
        #                      Enumerated Values: M = male F = female O = other.
        ds.PatientsBirthDate = '19010101'
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage

        # Study Instance UID tag 0x0020,0x000D (type UI - Unique Identifier)
        # self._dicom_study_instance_uid may be either set in __init__ when creating new object
        #   or set when import a DICOM file
        #   Study Instance UID for structures is the same as Study Instance UID for CTs
        ds.StudyInstanceUID = self._dicom_study_instance_uid

        # Series Instance UID tag 0x0020,0x000E (type UI - Unique Identifier)
        # self._pl_dicom_series_instance_uid may be either set in __init__ when creating new object
        #   Series Instance UID might be different than Series Instance UID for CTs
        ds.SeriesInstanceUID = self._plan_dicom_series_instance_uid

        ds.Modality = "RTPLAN"
        ds.SeriesDescription = 'RT Plan'
        ds.RTPlanDate = datetime.datetime.today().strftime('%Y%m%d')
        ds.RTPlanGeometry = ''
        ds.RTPlanLabel = 'B1'
        ds.RTPlanTime = datetime.datetime.today().strftime('%H%M%S')
        structure_ref = Dataset()
        structure_ref.RefdSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage
        structure_ref.RefdSOPInstanceUID = '1.2.3'
        ds.RefdStructureSets = Sequence([structure_ref])

        dose_ref = Dataset()
        dose_ref.DoseReferenceNumber = 1
        dose_ref.DoseReferenceStructureType = 'SITE'
        dose_ref.DoseReferenceType = 'TARGET'
        dose_ref.TargetPrescriptionDose = self.target_dose
        dose_ref.DoseReferenceDescription = "TUMOR"
        ds.DoseReferenceSequence = Sequence([dose_ref])
        return ds

    def write_dicom(self, directory):
        """ Write Dose-data to disk, in DICOM format.

        This file will save the dose cube and a plan associated with that dose.
        Function call create_dicom() and create_dicom_plan() and then save these.

        :param str directory: Directory where 'rtdose.dcm' and 'rtplan.dcm' will be stored.
        """
        dcm = self.create_dicom()
        dcm.save_as(os.path.join(directory, "rtdose.dcm"), write_like_original=False)
        # TODO add support for saving the plan
        # plan = self.create_dicom_plan()
        # plan.save_as(os.path.join(directory, "rtplan.dcm"))
