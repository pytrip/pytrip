import datetime
import logging

import numpy as np

from pydicom import Dataset, uid, FileDataset
from pydicom.tag import Tag

from pytrip.dicom.common import AccompanyingDicomData


def create_dicom_base(patient_name, patient_id, dimx, dimy, slice_distance, pixel_size):
    ds = Dataset()
    ds.PatientName = patient_name
    if patient_id in (None, ''):
        ds.PatientID = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
    else:
        ds.PatientID = patient_id  # Patient ID tag 0x0010,0x0020 (type LO - Long String)
    ds.PatientSex = 'O'  # Patient's Sex tag 0x0010,0x0040 (type CS - Code String)
    #                      Enumerated Values: M = male F = female O = other.
    ds.PatientBirthDate = '19010101'
    ds.SpecificCharacterSet = 'ISO_IR 100'
    ds.AccessionNumber = ''
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.SOPClassUID = '1.2.3'  # !!!!!!!!

    # Study Instance UID tag 0x0020,0x000D (type UI - Unique Identifier)
    ds.FrameofReferenceUID = '1.2.3'  # !!!!!!!!!
    ds.StudyDate = datetime.datetime.today().strftime('%Y%m%d')
    ds.StudyTime = datetime.datetime.today().strftime('%H%M%S')
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.SamplesPerPixel = 1
    ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']
    ds.Rows = dimx
    ds.Columns = dimy
    ds.SliceThickness = str(slice_distance)
    ds.PixelSpacing = [pixel_size, pixel_size]

    # Add eclipse friendly IDs
    ds.StudyID = '1'  # Study ID tag 0x0020,0x0010 (type SH - Short String)
    ds.ReferringPhysiciansName = 'py^trip'  # Referring Physician's Name tag 0x0008,0x0090 (type PN - Person Name)
    ds.PositionReferenceIndicator = ''  # Position Reference Indicator tag 0x0020,0x1040
    ds.SeriesNumber = '1'  # SeriesNumber tag 0x0020,0x0011 (type IS - Integer String)

    return ds


def create_dicom(cube, include_pixel_data=True):
    """ Creates a Dicom object from CTX cube.

    This function can be used to convert a TRiP98 CTX file to Dicom format.

    :returns: A Dicom object.
    """

    ds = create_dicom_base(cube.patient_name, cube.patient_id, cube.dimx, cube.dimy, cube.slice_distance,
                           cube.pixel_size)
    ds.Modality = 'CT'
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = cube.num_bytes * 8
    ds.BitsStored = cube.num_bytes * 8
    ds.HighBit = cube.num_bytes * 8 - 1
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

    # Manufacturer of the equipment that produced the composite instances.
    ds.Manufacturer = cube.creation_info  # Manufacturer tag, 0x0008,0x0070 (type LO - Long String)
    ds.KVP = ''  # KVP tag 0x0018, 0x0060

    ds.AcquisitionNumber = '1'  # AcquisitionNumber tag 0x0020, 0x0012 (type IS - Integer String)

    dicom_data = getattr(cube, 'dicom_data', {})
    headers_datasets = getattr(dicom_data, 'headers_datasets', {})
    all_ct_header_datasets = headers_datasets.get(AccompanyingDicomData.DataType.CT, {})
    data_datasets = getattr(dicom_data, 'data_datasets', {})
    all_ct_data_datasets = data_datasets.get(AccompanyingDicomData.DataType.CT, {})

    logging.debug("all_ct_header_datasets len {:d}".format(len(all_ct_header_datasets)))
    logging.debug("all_ct_header_datasets keys {:s}".format(",".join(all_ct_header_datasets.keys())))

    logging.debug("all_ct_data_datasets len {:d}".format(len(all_ct_data_datasets)))
    logging.debug("all_ct_data_datasets keys {:s}".format(",".join(all_ct_data_datasets.keys())))

    data = []  # list of DICOM objects with data specific to the slice

    meta = Dataset()
    meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    meta.ImplementationClassUID = "1.2.3.4"
    meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian  # Implicit VR Little Endian - Default Transfer Syntax

    for i in range(len(cube.cube)):

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
            _ds.SliceLocation = str(cube.slice_pos[i])

        if Tag('ImagePositionPatient') in current_ct_data_dataset:
            _ds.ImagePositionPatient = current_ct_data_dataset.ImagePositionPatient
        else:
            _ds.ImagePositionPatient = ["{:.3f}".format(cube.xoffset),
                                        "{:.3f}".format(cube.yoffset),
                                        "{:.3f}".format(cube.slice_pos[i])]

        if include_pixel_data:
            pixel_array_tmp = np.subtract(cube.cube[i][:][:], _ds.RescaleIntercept, casting='safe')
            pixel_array_tmp /= _ds.RescaleSlope
            pixel_array = pixel_array_tmp.astype(cube.pydata_type)
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
