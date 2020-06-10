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
import logging

import numpy as np

from pydicom import Dataset, FileDataset
from pydicom.tag import Tag

from pytrip.error import InputError
from pytrip.cube import Cube, AccompanyingDicomData

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
        print("Input Raw Pixel", dcm["images"][0].pixel_array.flatten()[0])

        for i in range(len(dcm["images"])):
            intersect = float(dcm["images"][i].RescaleIntercept)
            slope = float(dcm["images"][i].RescaleSlope)
            data = np.array(dcm["images"][i].pixel_array) * slope + intersect
            self.cube[i][:][:] = data
        if len(self.slice_pos) > 1 and self.slice_pos[1] < self.slice_pos[0]:
            self.slice_pos.reverse()
            self.zoffset = self.slice_pos[0]
            self.cube = self.cube[::-1]
        print("Input Cube Pixel", self.cube.flatten()[0])

    def create_dicom(self, include_pixel_data=True):
        """ Creates a Dicom object from self.

        This function can be used to convert a TRiP98 CTX file to Dicom format.

        :returns: A Dicom object.
        """

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

        dicom_data = getattr(self, 'dicom_data', {})
        headers_datasets = getattr(dicom_data, 'headers_datasets', {})
        all_ct_header_datasets = headers_datasets.get(AccompanyingDicomData.DataType.CT, {})
        data_datasets = getattr(dicom_data, 'data_datasets', {})
        all_ct_data_datasets = data_datasets.get(AccompanyingDicomData.DataType.CT, {})

        logging.debug("all_ct_header_datasets len {:d}".format(len(all_ct_header_datasets)))
        logging.debug("all_ct_header_datasets keys {:s}".format(",".join(all_ct_header_datasets.keys())))

        logging.debug("all_ct_data_datasets len {:d}".format(len(all_ct_data_datasets)))
        logging.debug("all_ct_data_datasets keys {:s}".format(",".join(all_ct_data_datasets.keys())))

        from pydicom import uid
        data = []  # list of DICOM objects with data specific to the slice

        meta = Dataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        meta.ImplementationClassUID = "1.2.3.4"
        meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian  # Implicit VR Little Endian - Default Transfer Syntax

        for i in range(len(self.cube)):

            # a copy of dataset
            _ds = Dataset()
            _ds.update(ds)

            _ds.InstanceNumber = str(i + 1)

            current_ct_header_dataset = all_ct_header_datasets.get(str(_ds.InstanceNumber), {})
            current_ct_data_dataset = all_ct_data_datasets.get(str(_ds.InstanceNumber), {})

            # overwrite some tags if the cube has some DICOM data stored (i.e. was previously imported from DICOM data)

            for tag in ['AcquisitionDate', 'AcquisitionDateTime', 'AcquisitionNumber', 'AcquisitionTime', 'BitsStored',
                        'BodyPartExamined', 'CTDIPhantomTypeCodeSequence', 'CTDIvol', 'CalciumScoringMassFactorDevice',
                        'CalciumScoringMassFactorDevice', 'ContentDate', 'ContentTime', 'ConvolutionKernel',
                        'DataCollectionCenterPatient', 'DataCollectionCenterPatient', 'DataCollectionCenterPatient',
                        'DataCollectionDiameter', 'DateOfLastCalibration', 'DeviceSerialNumber',
                        'DistanceSourceToDetector', 'DistanceSourceToPatient', 'EstimatedDoseSaving', 'Exposure',
                        'ExposureModulationType', 'ExposureTime', 'FilterType', 'FocalSpots', 'FrameOfReferenceUID',
                        'GantryDetectorTilt', 'GeneratorPower', 'HighBit', 'ImageComments', 'ImagePositionPatient',
                        'ImagePositionPatient', 'ImageType', 'InstitutionAddress', 'InstitutionName',
                        'IrradiationEventUID', 'KVP', 'LargestImagePixelValue', 'LargestImagePixelValue',
                        'Manufacturer', 'ManufacturerModelName', 'OtherPatientNames', 'PatientAge', 'PatientBirthDate',
                        'PatientID', 'PatientSex', 'PixelRepresentation', 'ProtocolName', 'ReconstructionDiameter',
                        'ReconstructionTargetCenterPatient', 'ReconstructionTargetCenterPatient',
                        'ReconstructionTargetCenterPatient', 'ReferencedImageSequence', 'ReferringPhysicianName',
                        'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'RotationDirection',
                        'SequenceDelimitationItem', 'SeriesDate', 'SeriesDescription', 'SeriesInstanceUID',
                        'SeriesNumber', 'SeriesTime', 'SingleCollimationWidth', 'SmallestImagePixelValue',
                        'SmallestImagePixelValue', 'SoftwareVersions', 'SourceImageSequence', 'SpiralPitchFactor',
                        'StationName', 'StudyDate', 'StudyDescription', 'StudyID', 'StudyInstanceUID', 'StudyTime',
                        'TableFeedPerRotation', 'TableHeight', 'TableSpeed', 'TimeOfLastCalibration',
                        'TotalCollimationWidth', 'WindowCenter', 'WindowCenterWidthExplanation', 'WindowWidth',
                        'XRayTubeCurrent']:
                if Tag(tag) in current_ct_data_dataset:
                    _ds[tag] = current_ct_data_dataset[tag]
            # SOP Instance UID tag 0x0008,0x0018 (type UI - Unique Identifier)
            if Tag('SOPInstanceUID') in current_ct_data_dataset:
                _ds.SOPInstanceUID = current_ct_data_dataset.SOPInstanceUID
            else:
                _ds.SOPInstanceUID = uid.generate_uid(prefix=None)

            if Tag('SliceLocation') in current_ct_data_dataset:
                _ds.SliceLocation = current_ct_data_dataset.SliceLocation
            else:
                _ds.SliceLocation = str(self.slice_pos[i])

            if Tag('ImagePositionPatient') in current_ct_data_dataset:
                _ds.ImagePositionPatient = current_ct_data_dataset.ImagePositionPatient
            else:
                _ds.ImagePositionPatient = ["{:.3f}".format(self.xoffset),
                                            "{:.3f}".format(self.yoffset),
                                            "{:.3f}".format(self.slice_pos[i])]

            if include_pixel_data:
                pixel_array_tmp = np.subtract(self.cube[i][:][:], _ds.RescaleIntercept, casting='safe')
                pixel_array_tmp /= _ds.RescaleSlope
                pixel_array = pixel_array_tmp.astype(self.pydata_type)
                _ds.PixelData = pixel_array.tostring()

            fds = FileDataset("file", _ds, file_meta=meta, preamble=b"\0" * 128)

            # Media Storage SOP Instance UID tag 0x0002,0x0003 (type UI - Unique Identifier)
            if Tag('MediaStorageSOPInstanceUID') not in fds.file_meta:
                fds.file_meta.MediaStorageSOPInstanceUID = _ds.SOPInstanceUID

            # overwrite some tags if the cube has some DICOM data stored (i.e. was previously imported from DICOM data)
            for tag in ['ImplementationClassUID', 'ImplementationVersionName', 'MediaStorageSOPInstanceUID']:
                if Tag(tag) in current_ct_header_dataset:
                    fds.file_meta[tag] = current_ct_header_dataset[tag]

            data.append(fds)

        return data

    def write_dicom(self, directory):
        """ Write CT-data to disk, in Dicom format.

        :param str directory: directory to write to. If directory does not exist, it will be created.
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        dcm_list = self.create_dicom()
        for dcm_item in dcm_list:
            output_filename = "CT.PYTRIP.{:d}.dcm".format(dcm_item.InstanceNumber)
            logger.info("Saving {}".format(output_filename))
            dcm_item.save_as(os.path.join(directory, output_filename))
