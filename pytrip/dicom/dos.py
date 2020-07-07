import numpy as np
from pydicom import Dataset, Sequence, uid, FileDataset
from pydicom._storage_sopclass_uids import RTIonPlanStorage
from pydicom.datadict import tag_for_keyword

from pytrip.dicom.common import AccompanyingDicomData
from pytrip.dicom.ct import create_dicom_base


def create_dicom(cube):
    """ Creates a DICOM RT-Dose object from self.

    This function can be used to convert a TRiP98 Dose file to DICOM format.

    :returns: a DICOM RT-Dose object.
    """

    dicom_data = getattr(cube, 'dicom_data', {})
    headers_datasets = getattr(dicom_data, 'headers_datasets', {})
    ct_header_dataset = headers_datasets.get(AccompanyingDicomData.DataType.CT, {})
    if ct_header_dataset:
        first_ct_header = ct_header_dataset.get(list(ct_header_dataset.keys())[0], {})
    else:
        first_ct_header = {}

    data_datasets = getattr(dicom_data, 'data_datasets', {})
    ct_data_dataset = data_datasets.get(AccompanyingDicomData.DataType.CT, {})
    if ct_data_dataset:
        first_ct_dataset = ct_data_dataset.get(list(ct_data_dataset.keys())[0], {})
    else:
        first_ct_dataset = {}

    ds = create_dicom_base(cube.patient_name, cube.patient_id, cube.dimx, cube.dimy, cube.slice_distance,
                           cube.pixel_size)
    ds.Modality = 'RTDOSE'

    ds.SamplesPerPixel = 1
    ds.BitsAllocated = cube.num_bytes * 8
    ds.BitsStored = ds.BitsAllocated
    ds.HighBit = ds.BitsStored - 1

    ds.AccessionNumber = ''
    ds.SeriesDescription = 'RT Dose'
    ds.DoseUnits = 'GY'
    ds.DoseType = 'PHYSICAL'

    if cube.pydata_type in {np.float32, np.float64}:
        ds.DoseGridScaling = 1.0
    else:
        ds.DoseGridScaling = cube.target_dose / 1000.0

    ds.DoseSummationType = 'PLAN'
    ds.SliceThickness = cube.slice_distance
    ds.InstanceCreationDate = '19010101'
    ds.InstanceCreationTime = '000000'
    ds.NumberOfFrames = len(cube.cube)
    ds.PixelRepresentation = 0
    ds.StudyID = '1'
    ds.SeriesNumber = '1'  # SeriesNumber tag 0x0020,0x0011 (type IS - Integer String)
    ds.GridFrameOffsetVector = [x * cube.slice_distance for x in range(cube.dimz)]
    ds.InstanceNumber = ''
    ds.PositionReferenceIndicator = "RF"
    ds.TissueHeterogeneityCorrection = ['IMAGE', 'ROI_OVERRIDE']
    ds.ImagePositionPatient = ["%.3f" % (cube.xoffset * cube.pixel_size), "%.3f" % (cube.yoffset * cube.pixel_size),
                               "%.3f" % (cube.slice_pos[0])]
    ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'
    ds.SOPInstanceUID = cube._dose_dicom_SOP_instance_uid

    # Study Instance UID tag 0x0020,0x000D (type UI - Unique Identifier)
    # self._dicom_study_instance_uid may be either set in __init__ when creating new object
    #   or set when import a DICOM file
    #   Study Instance UID for structures is the same as Study Instance UID for CTs
    # ds.StudyInstanceUID = self._dicom_study_instance_uid

    # Series Instance UID tag 0x0020,0x000E (type UI - Unique Identifier)
    # self._dose_dicom_series_instance_uid may be either set in __init__ when creating new object
    #   Series Instance UID might be different than Series Instance UID for CTs
    ds.SeriesInstanceUID = cube._dose_dicom_series_instance_uid

    # Bind to rtplan
    rt_set = Dataset()
    rt_set.ReferencedSOPInstanceUID = cube._plan_dicom_series_instance_uid
    rt_set.ReferencedSOPClassUID = RTIonPlanStorage
    ds.ReferencedRTPlanSequence = Sequence([rt_set])
    pixel_array = np.zeros((len(cube.cube), ds.Rows, ds.Columns), dtype=cube.pydata_type)

    tags_to_be_imported = [tag_for_keyword(name) for name in
                           ['StudyDate', 'StudyTime', 'StudyDescription', 'ImageOrientationPatient',
                            'Manufacturer', 'ReferringPhysicianName', 'Manufacturer', 'ImagePositionPatient',
                            'FrameOfReferenceUID', 'PositionReferenceIndicator', 'SeriesNumber', 'StudyInstanceUID',
                            'PatientBirthDate', 'PatientID', 'PatientName']]

    ct_datasets_data_common = getattr(dicom_data, 'ct_datasets_data_common', {})
    for tag_number, _ in ct_datasets_data_common:
        if tag_number in tags_to_be_imported:
            ds[tag_number] = first_ct_dataset[tag_number]

    patient_positions = []
    for ct_ds in ct_data_dataset.values():
        if 'ImagePositionPatient' in ct_ds:
            patient_positions.append(ct_ds.ImagePositionPatient)
    if patient_positions:
        ds.ImagePositionPatient = patient_positions[0]
        ds.ImagePositionPatient[2] = min(pos[2] for pos in patient_positions)

    pixel_array[:][:][:] = cube.cube[:][:][:]
    ds.PixelData = pixel_array.tostring()

    meta = Dataset()
    meta.MediaStorageSOPClassUID = RTIonPlanStorage
    meta.MediaStorageSOPInstanceUID = cube._dose_dicom_SOP_instance_uid
    meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian  # Implicit VR Little Endian - Default Transfer Syntax

    tags_to_be_imported = [tag_for_keyword(name) for name in
                           ['ImplementationClassUID', 'ImplementationVersionName']]
    ct_datasets_header_common = getattr(dicom_data, 'ct_datasets_header_common', {})
    for tag_number in ct_datasets_header_common:
        if tag_number in tags_to_be_imported:
            ds[tag_number] = first_ct_header[tag_number]

    fds = FileDataset("file", ds, file_meta=meta, preamble=b"\0" * 128)

    return fds
