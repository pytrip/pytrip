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
This module provides the DosCube class, which the user should use for handling plan-generated dose distributions.
"""
import os
import logging

import numpy as np

from pytrip.error import InputError, ModuleNotLoadedError
from pytrip.cube import Cube
import pytriplib

try:
    from dicom.dataset import Dataset, FileDataset
    from dicom.sequence import Sequence
    from dicom import UID

    _dicom_loaded = True
except:
    _dicom_loaded = False

logger = logging.getLogger(__name__)


class DosCube(Cube):
    """ Class for handling Dose data. In TRiP98 these are stored in VOXELPLAN format with the .dos suffix.
    This class can also handle Dicom files.
    """
    data_file_extension = "dos"

    def __init__(self, cube=None):
        super(DosCube, self).__init__(cube)
        self.type = "DOS"
        self.target_dose = 0.0

    def read_dicom(self, dcm):
        """ Imports the dose distribution from Dicom object.

        :param Dicom dcm: a Dicom object
        """
        if "rtdose" not in dcm:
            raise InputError("Data doesn't contain dose infomation")
        if self.header_set is False:
            self.read_dicom_header(dcm)
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
                dose_bins += pytriplib.calculate_dvh_slice(self.cube[i], np.array(slice.contour[0].contour), voxel_size)

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
        dvh_tuple = self.calculate_dvh(voi)
        if dvh_tuple is None:
            logger.warning("Voi {:s} outside the cube".format(voi.get_name()))
        else:
            dvh, min_dose, max_dose, mean_dose, mean_volume = dvh_tuple
            np.savetxt(dvh, filename)

    def create_dicom_plan(self):
        """ Create a dummy Dicom RT-plan object.

        The only data which is forwarded to this object, is self.patient_name.
        :returns: a Dicom RT-plan object.
        """
        meta = Dataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        meta.MediaStorageSOPInstanceUID = "1.2.3"
        meta.ImplementationClassUID = "1.2.3.4"
        meta.TransferSyntaxUID = UID.ImplicitVRLittleEndian  # Implicit VR Little Endian - Default Transfer Syntax
        ds = FileDataset("file", {}, file_meta=meta, preamble=b"\0" * 128)
        ds.PatientsName = self.patient_name
        ds.PatientID = "123456"
        ds.PatientsSex = '0'
        ds.PatientsBirthDate = '19010101'
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        ds.StudyInstanceUID = '1.2.3'
        ds.SOPInstanceUID = '1.2.3'

        ds.Modality = "RTPLAN"
        ds.SeriesDescription = 'RT Plan'
        ds.SeriesInstanceUID = '2.16.840.1.113662.2.12.0.3057.1241703565.43'
        ds.RTPlanDate = '19010101'
        ds.RTPlanGeometry = ''
        ds.RTPlanLabel = 'B1'
        ds.RTPlanTime = '000000'
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
        ds.DoseReferences = Sequence([dose_ref])
        return ds

    def create_dicom(self):
        """ Creates a Dicom RT-Dose object from self.

        This function can be used to convert a TRiP98 Dose file to Dicom format.

        :returns: a Dicom RT-Dose object.
        """

        if not _dicom_loaded:
            raise ModuleNotLoadedError("Dicom")
        if not self.header_set:
            raise InputError("Header not loaded")

        ds = self.create_dicom_base()
        ds.Modality = 'RTDOSE'
        ds.SamplesperPixel = 1
        ds.BitsAllocated = self.num_bytes * 8
        ds.BitsStored = self.num_bytes * 8
        ds.AccessionNumber = ''
        ds.SeriesDescription = 'RT Dose'
        ds.DoseUnits = 'GY'
        ds.DoseType = 'PHYSICAL'
        ds.DoseGridScaling = self.target_dose / 10**5
        ds.DoseSummationType = 'PLAN'
        ds.SliceThickness = ''
        ds.InstanceCreationDate = '19010101'
        ds.InstanceCreationTime = '000000'
        ds.NumberOfFrames = len(self.cube)
        ds.PixelRepresentation = 0
        ds.StudyID = '1'
        ds.SeriesNumber = 14
        ds.GridFrameOffsetVector = [x * self.slice_distance for x in range(self.dimz)]
        ds.InstanceNumber = ''
        ds.NumberofFrames = len(self.cube)
        ds.PositionReferenceIndicator = "RF"
        ds.TissueHeterogeneityCorrection = ['IMAGE', 'ROI_OVERRIDE']
        ds.ImagePositionPatient = ["%.3f" % (self.xoffset * self.pixel_size), "%.3f" % (self.yoffset * self.pixel_size),
                                   "%.3f" % (self.slice_pos[0])]
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'
        ds.SOPInstanceUID = '1.2.246.352.71.7.320687012.47206.20090603085223'
        ds.SeriesInstanceUID = '1.2.246.352.71.2.320687012.28240.20090603082420'

        # Bind to rtplan
        rt_set = Dataset()
        rt_set.RefdSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'
        rt_set.RefdSOPInstanceUID = '1.2.3'
        ds.ReferencedRTPlans = Sequence([rt_set])
        pixel_array = np.zeros((len(self.cube), ds.Rows, ds.Columns), dtype=self.pydata_type)
        pixel_array[:][:][:] = self.cube[:][:][:]
        ds.PixelData = pixel_array.tostring()
        return ds

    def write(self, path):
        """
        Write Dose data to disk, in TRiP98/Voxelplan format.

        This method will build and write both the .hed and .dos file.

        :param str path: Path, any file extentions will be ignored.
        """
        f_split = os.path.splitext(path)
        header_file = f_split[0] + ".hed"
        dos_file = f_split[0] + ".dos"
        self.write_trip_header(header_file)
        self.write_trip_data(dos_file)

    def write_dicom(self, directory):
        """ Write Dose-data to disk, in Dicom format.

        This file will save the dose cube and a plan associated with that dose.
        Function call create_dicom() and create_dicom_plan() and then save these.

        :param str directory: Directory where 'rtdose.dcm' and 'trplan.dcm' will be stored.
        """
        dcm = self.create_dicom()
        plan = self.create_dicom_plan()
        dcm.save_as(os.path.join(directory, "rtdose.dcm"))
        plan.save_as(os.path.join(directory, "rtplan.dcm"))
