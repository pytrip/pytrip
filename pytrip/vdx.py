#
#    Copyright (C) 2010-2018 PyTRiP98 Developers.
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
This module holds all the user needs to deal with Volume of interests.
It provides the top-level VdxCube class, Voi, Slice and Contour classes.
The Voi class represents a volume of interest 'VOI', also called region of interest 'ROI' in Dicom lingo.
Each Voi holds several Slice, which are normally synced with an associated CT-cube.
Each Slice holds one or more Contours.
"""
import os
import copy
from math import pi, sqrt
import logging
import warnings
from functools import cmp_to_key

import numpy as np

try:
    # as of version 1.0 pydicom package import has beed renamed from dicom to pydicom
    from pydicom import uid
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.sequence import Sequence
    _dicom_loaded = True
except ImportError:
    try:
        # fallback to old (<1.0) pydicom package version
        from dicom import UID as uid  # old pydicom had UID instead of uid
        from dicom.dataset import Dataset, FileDataset
        from dicom.sequence import Sequence
        _dicom_loaded = True
    except ImportError:
        _dicom_loaded = False

import pytrip
from pytrip.error import InputError, ModuleNotLoadedError
from pytrip.dos import DosCube
import pytriplib

logger = logging.getLogger(__name__)


class VdxCube:
    """
    VdxCube is the master class for dealing with Volume of Interests (VOIs).
    A VdxCube contains one or more VOIs which are structures which represent
    some organ (lung, eye ...) or target (GTV, PTV...)
    The Voi object contains Slice objects which corresponds to the CT slices,
    and the slice objects contains contour objects.
    Each contour object are a set of points which delimit a closed region.
    One single slice object can contain multiple contours.

    VdxCube ---> Voi[] ---> Slice[] ---> Contour[] ---> Point[]

    Note, since TRiP98 only supports one contour per slice for each voi.
    PyTRiP supports functions for connecting multiple contours to a single
    entity using infinite thin connects.

    VdxCube can import both dicom data and TRiP data,
    and export in the those formats.

    We strongly recommend to load a CT and/or a DOS cube first, see example below:

    >>> import pytrip as pt
    >>> c = pt.CtxCube()
    >>> c.read("tests/res/TST003/tst003000.ctx.gz")
    >>> v = pt.VdxCube(c)
    >>> v.read("tests/res/TST003/tst003000.vdx")
    """

    # stictly, VDX does not have .hed companion. However, in practice, .vdx files are always
    # associated with a .hed .ctx cube pair. In order to discover these, the header file
    # extentions may be associated to this as a .vdx data file.

    header_file_extension = '.hed'
    data_file_extension = '.vdx'
    allowed_suffix = tuple()

    def __init__(self, cube=None):
        """
        :param cube: CtxCube type object
        """
        self.vois = []
        self.cube = cube
        self.path = ""  # full path to .vdx file, set if a regular .vdx file was loaded loaded

        # UIDs unique for whole structure set
        # generation of UID is done here in init, the reason why we are not generating them in create_dicom
        # method is that subsequent calls to write method shouldn't changed UIDs
        self._dicom_study_instance_uid = uid.generate_uid(prefix=None)
        self._structs_dicom_series_instance_uid = uid.generate_uid(prefix=None)
        self._structs_sop_instance_uid = uid.generate_uid(prefix=None)
        self._structs_rt_series_instance_uid = uid.generate_uid(prefix=None)

        self.version = "2.0"
        if self.cube is not None:
            self.patient_id = cube.patient_id
            self._dicom_study_instance_uid = self.cube._dicom_study_instance_uid
            logger.debug("VDX class inherited patient_id {}".format(self.patient_id))
        else:
            import datetime
            self.patient_id = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
            logger.debug("VDX class creates new patient_id {}".format(self.patient_id))

    def __str__(self):
        """ str output handler
        """
        return self._print()

    def _print(self):
        """ Pretty print current attributes.
        """
        out = "\n"
        out += "   VdxCube\n"
        out += "----------------------------------------------------------------------------\n"
        out += "| UIDs\n"
        out += "|   dicom_study_instance_uid            : {:s}\n".format(self._dicom_study_instance_uid)
        out += "|   structs_dicom_series_instance_uid   : '{:s}'\n".format(self._structs_dicom_series_instance_uid)
        out += "|   structs_sop_instance_uid            : '{:s}'\n".format(self._structs_sop_instance_uid)
        out += "|   structs_rt_series_instance_uid      : '{:s}'\n".format(self._structs_rt_series_instance_uid)
        if self.vois:
            out += "+---VOIs\n"
            for _i, _v in enumerate(self.vois):
                out += "|   |           #{:d}              : '{:s}'\n".format(_i, _v.name)
        return out

    def read_dicom(self, data, structure_ids=None):
        """
        Reads structures from a Dicom RTSS Object.

        :param Dicom data: A Dicom RTSS object.
        :param structure_ids: (TODO: undocumented)
        """
        if "rtss" not in data:
            raise InputError("Input is not a valid rtss structure")
        dcm = data["rtss"]
        self.version = "2.0"

        self._dicom_study_instance_uid = dcm.StudyInstanceUID
        self._structs_dicom_series_instance_uid = dcm.SeriesInstanceUID
        self._structs_sop_instance_uid = dcm.SOPInstanceUID

        self.basename = dcm.PatientID.replace(" ", "_")

        if hasattr(dcm, 'ROIContourSequence'):
            _contours = dcm.ROIContourSequence
        else:
            logger.error("No ROIContours or ROIContourSequence found in dicom RTSTRUCT")
            exit()

        for i, _roi_contour in enumerate(_contours):
            if structure_ids is None or _roi_contour.RefdROINumber in structure_ids:
                if hasattr(dcm, "RTROIObservationsSequence"):
                    _roi_observation = dcm.RTROIObservationsSequence[i]
                    # _roi_name = dcm.RTROIObservationsSequence[i].ROIObservationLabel  # OPTIONAL by DICOM
                    # kept for future use.

                else:
                    logger.error("No RTROIObservations or RTROIObservationsSequence found in dicom RTSTRUCT")
                    exit()

                if hasattr(dcm, "StructureSetROISequence"):
                    _roi_name = dcm.StructureSetROISequence[i].ROIName  # REQUIRED by DICOM. At least an empty string.
                else:
                    logger.error("No StructureSetROISequence found in dicom RTSTRUCT")
                    exit()

                v = Voi(_roi_name, self.cube)
                v.read_dicom(_roi_observation, _roi_contour, _roi_name)

                self.add_voi(v)

    def get_voi_names(self):
        """
        :returns: a list of available voi names.
        :warning: deprecated method, use voi_names() instead.
        """
        warnings.warn("Call to deprecated method get_voi_names(). Use voi_names() instead.",
                      category=DeprecationWarning,
                      stacklevel=2)
        return self.voi_names()

    def voi_names(self):
        """
        :returns: a list of available voi names.
        """
        names = [voi.name for voi in self.vois]
        return names

    def add_voi(self, voi):
        """ Appends a new voi to this class.

        :param Voi voi: the voi to be appended to this class.
        """
        self.vois.append(voi)

    def get_voi_by_name(self, name):
        """ Returns a Voi object by its name.

        :param str name: Name of voi to be returned.
        :returns: the Voi which has exactly this name, else raise an Error.
        """
        for voi in self.vois:
            if voi.name.lower() == name.lower():
                return voi
        raise InputError("Voi doesn't exist")

    def import_vdx(self, path):
        """ Reads a structure file in Voxelplan format.
        This method is identical to self.read() and self.read_vdx()

        :param str path: Full path including file extension.
        """
        self.read_vdx(path)

    def read(self, path):
        """ Reads a structure file in Voxelplan format.
        This method is identical to self.import_vdx() and self.read_vdx()

        :param str path: Full path including file extension.
        """
        self.read_vdx(path)

    def vdx_version(self, content):
        """ Test content for what VDX version this is.
        Since the VDX version is very undocumented, we are guessing here.

        :param content: the vdx file contents as an array of strings.
        :returns: vdx version string, e.g. "1.2" or "2.0"
        """
        for line in content:
            if line.strip().startswith("vdx_file_version"):
                return line.split()[1]
            else:
                return "1.2"

    def read_vdx(self, path):
        """ Reads a structure file in Voxelplan format.

        :param str path: Full path including file extension.
        """
        self.basename = os.path.basename(path).split(".")[0]
        self.path = path
        fp = open(path, "r")
        content = fp.read().split('\n')
        fp.close()

        self.version = self.vdx_version(content)

        i = 0
        n = len(content)
        header_full = False
        #        number_of_vois = 0
        while i < n:
            line = content[i].strip()
            if not header_full:
                if line.startswith("all_indices_zero_based"):
                    self.zero_based = True
#                TODO number_of_vois not used
#                elif "number_of_vois" in line:
#                    number_of_vois = int(line.split()[1])
            if line.startswith("voi"):
                v = Voi(line.split()[1], self.cube)
                if self.version == "1.2":
                    _token = line.split()
                    if len(_token) == 6:
                        if _token[5] != '0':
                            i = v.read_vdx_old(content, i)
                else:
                    i = v.read_vdx(content, i)
                self.add_voi(v)
                header_full = True
            i += 1

    def concat_contour(self):
        """ Loop through all available VOIs and check whether any have multiple contours in a slice.
        If so, merge them to a single contour.

        This is needed since TRiP98 cannot handle multiple contours in the same slice.
        """
        for i in range(len(self.vois)):
            self.vois[i].concat_contour()

    def number_of_vois(self):
        """
        :returns: the number of VOIs in this object.
        """
        return len(self.vois)

    def _write_vdx(self, path):
        """ Writes all VOIs in voxelplan format.

        All will be written in version 2.0 format, irrespectively of self.version.

        :param str path: Full path, including file extension (.vdx).
        """
        fp = open(path, "w")
        fp.write("vdx_file_version 2.0\n")
        fp.write("all_indices_zero_based\n")
        fp.write("number_of_vois {:d}\n".format(self.number_of_vois()))
        self.vois = sorted(self.vois, key=lambda voi: voi.type, reverse=True)
        for voi in self.vois:
            logger.debug("writing VOI {}".format(voi.name))
            fp.write(voi.vdx_string())
        fp.close()

    def write_trip(self, path):
        """ Writes all VOIs in voxelplan format, while ensuring no slice holds more than one contour.
        Identical to write().

        :param str path: Full path, including file extension (.vdx).
        """
        self.concat_contour()
        self._write_vdx(path)

    def write(self, path):
        """ Writes all VOIs in voxelplan format, while ensuring no slice holds more than one contour.
        Identical to write_trip().

        :param str path: Full path, including file extension (.vdx).
        """
        self.write_trip(path)

    def create_dicom(self):
        """ Generates and returns Dicom RTSTRUCT object, which holds all VOIs.

        :returns: a Dicom RTSTRUCT object holding any VOIs.
        """
        if _dicom_loaded is False:
            raise ModuleNotLoadedError("Dicom")
        meta = Dataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage SOP Class
        # SOP Instance UID tag 0x0002,0x0003 (type UI - Unique Identifier)
        meta.MediaStorageSOPInstanceUID = self._structs_sop_instance_uid
        meta.ImplementationClassUID = "1.2.3.4"
        meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian  # Implicit VR Little Endian - Default Transfer Syntax
        ds = FileDataset("file", {}, file_meta=meta, preamble=b"\0" * 128)
        if self.cube is not None:
            ds.PatientName = self.cube.patient_name
            ds.Manufacturer = self.cube.creation_info  # Manufacturer tag, 0x0008,0x0070 (type LO - Long String)
        else:
            ds.PatientName = ''
            ds.Manufacturer = ''  # Manufacturer tag, 0x0008,0x0070 (type LO - Long String)
        ds.SeriesNumber = '1'  # SeriesNumber tag 0x0020,0x0011 (type IS - Integer String)
        ds.PatientID = self.patient_id  # patient_id of the VdxCube, from CtxCube during __init__.
        ds.PatientSex = ''  # Patient's Sex tag 0x0010,0x0040 (type CS - Code String)
        #                      Enumerated Values: M = male F = female O = other.
        ds.PatientBirthDate = '19010101'
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.AccessionNumber = ''
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage SOP Class

        # SOP Instance UID tag 0x0008,0x0018 (type UI - Unique Identifier)
        ds.SOPInstanceUID = self._structs_sop_instance_uid

        # Study Instance UID tag 0x0020,0x000D (type UI - Unique Identifier)
        # self._dicom_study_instance_uid may be either set in __init__ when creating new object
        #   or set when import a DICOM file
        #   Study Instance UID for structures is the same as Study Instance UID for CTs
        ds.StudyInstanceUID = self._dicom_study_instance_uid

        # Series Instance UID tag 0x0020,0x000E (type UI - Unique Identifier)
        # self._rt_dicom_series_instance_uid may be either set in __init__ when creating new object
        #   or set when import a DICOM file
        #   Series Instance UID for structures might be different than Series Instance UID for CTs
        ds.SeriesInstanceUID = self._structs_dicom_series_instance_uid

        ds.FrameOfReferenceUID = '1.2.3'  # !!!!!!!!!
        ds.SeriesDate = '19010101'  # !!!!!!!!
        ds.ContentDate = '19010101'  # !!!!!!
        ds.StudyDate = '19010101'  # !!!!!!!
        ds.StudyID = '1'  # Study ID tag 0x0020,0x0010 (type SH - Short String)
        ds.SeriesTime = '000000'  # !!!!!!!!!
        ds.StudyTime = '000000'  # !!!!!!!!!!
        ds.ContentTime = '000000'  # !!!!!!!!!
        ds.StructureSetLabel = 'PyTRiP structs'  # Structure set label tag, 0x3006,0x0002 (type SH - Short String)
        # Short string (SH) is limited to 16 characters !
        ds.StructureSetDate = '19010101'
        ds.StructureSetTime = '000000'
        ds.StructureSetName = 'ROI'
        ds.Modality = 'RTSTRUCT'
        ds.ROIGenerationAlgorithm = '0'  # ROI Generation Algorithm tag, 0x3006,0x0036 (type CS - Code String)
        ds.ReferringPhysicianName = 'py^trip'  # Referring Physician's Name tag 0x0008,0x0090 (type PN - Person Name)

        roi_label_list = []
        roi_data_list = []
        roi_structure_roi_list = []

        # to get DICOM which can be loaded in Eclipse we need to store information about UIDs of all slices in CT
        # first we check if DICOM cube is loaded
        if self.cube is not None:
            rt_ref_series_data = Dataset()
            rt_ref_series_data.SeriesInstanceUID = self.cube._ct_dicom_series_instance_uid
            rt_ref_series_data.ContourImageSequence = Sequence([])

            # each CT slice corresponds to one DICOM file
            for slice_dicom in self.cube.create_dicom():
                slice_dataset = Dataset()
                slice_dataset.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage SOP Class
                slice_dataset.ReferencedSOPInstanceUID = slice_dicom.SOPInstanceUID  # most important - slice UID
                rt_ref_series_data.ContourImageSequence.append(slice_dataset)

            rt_ref_study_seq_data = Dataset()
            rt_ref_study_seq_data.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'  # Study Component Management Class
            rt_ref_study_seq_data.ReferencedSOPInstanceUID = '1.2.3'
            rt_ref_study_seq_data.RTReferencedSeriesSequence = Sequence([rt_ref_series_data])

            rt_ref_frame_study_data = Dataset()
            rt_ref_frame_study_data.RTReferencedStudySequence = Sequence([rt_ref_study_seq_data])
            rt_ref_frame_study_data.FrameOfReferenceUID = '1.2.3'
            # (3006, 0010) 'Referenced Frame of Reference Sequence'
            ds.ReferencedFrameOfReferenceSequence = Sequence([rt_ref_frame_study_data])

        for i in range(self.number_of_vois()):
            logger.debug("Write ROI #{:d} to DICOM object".format(i))
            roi_label = self.vois[i].create_dicom_label()
            roi_label.ObservationNumber = str(i + 1)
            roi_label.ReferencedROINumber = str(i + 1)
            roi_label.RefdROINumber = str(i + 1)
            roi_contours = self.vois[i].create_dicom_contour_data(i)
            roi_contours.RefdROINumber = str(i + 1)
            roi_contours.ReferencedROINumber = str(i + 1)

            roi_structure_roi = self.vois[i].create_dicom_structure_roi()
            roi_structure_roi.ROINumber = str(i + 1)

            # (3006, 0024) Referenced Frame of Reference UID   (UI)
            roi_structure_roi.ReferencedFrameOfReferenceUID = rt_ref_frame_study_data.FrameOfReferenceUID
            roi_structure_roi_list.append(roi_structure_roi)
            roi_label_list.append(roi_label)
            roi_data_list.append(roi_contours)
        ds.RTROIObservations = Sequence(roi_label_list)
        ds.ROIContours = Sequence(roi_data_list)
        ds.StructureSetROIs = Sequence(roi_structure_roi_list)
        return ds

    def write_dicom(self, directory):
        """ Generates a Dicom RTSTRUCT object from self, and writes it to disk.

        :param str directory: Directory where the rtss.dcm file will be saved.
        """
        dcm = self.create_dicom()
        dcm.save_as(os.path.join(directory, "RTSTRUCT.PYTRIP.dcm"))


def _voi_point_cmp(a, b):
    """ TODO: needs documentation """
    if abs(a[1] - b[1]) < 0.2:
        c = a[0] - b[0]
    else:
        c = a[1] - b[1]
    if c < 0:
        return -1
    else:
        return 1


def create_cube(cube, name, center, width, height, depth):
    """
    Creates a new VOI which holds the contours rendering a square box

    :param Cube cube: A CTX or DOS cube to work on.
    :param str name: Name of the VOI
    :param [float*3] center: Center position [x,y,z] in [mm]
    :param float width: Width of box, along x in [mm]
    :param float height: Height of box, along y in [mm]
    :param float depth: Depth of box, along z in [mm]
    :returns: A new Voi object.
    """
    v = Voi(name, cube)
    for i in range(0, cube.dimz):
        z = i * cube.slice_distance
        if center[2] - depth / 2 <= z <= center[2] + depth / 2:
            s = Slice(cube)
            s.thickness = cube.slice_distance
            points = [  # 4 corners of cube in this slice
                [center[0] - width / 2, center[1] - height / 2, z],
                [center[0] + width / 2, center[1] - height / 2, z],
                [center[0] + width / 2, center[1] + height / 2, z],
                [center[0] - width / 2, center[1] + height / 2, z]
            ]
            c = Contour(points, cube)
            c.contour_closed = True
            s.add_contour(c)
            v.add_slice(s)
    return v


def create_voi_from_cube(cube, name, value=100):
    """
    Creates a new VOI which holds the contours following an isodose lines.

    :param Cube cube: A CTX or DOS cube to work on.
    :param str name: Name of the VOI
    :param int value: The isodose value from which the contour will be generated from.
    :returns: A new Voi object.
    """
    v = Voi(name, cube)
    # there are some cases when this script is run on systems without DISPLAY variable being set
    # in such case matplotlib backend has to be explicitly specified
    # we do it here and not in the top of the file, as interleaving imports with code lines is discouraged
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib._cntr as cntr
    for i in range(cube.dimz):
        x, y = np.meshgrid(np.arange(len(cube.cube[0, 0])), np.arange(len(cube.cube[0])))
        isodose_obj = cntr.Cntr(x, y, cube.cube[i])
        contour = isodose_obj.trace(value)
        s = Slice(cube)
        s.thickness = cube.slice_distance
        if not len(contour):
            continue
        points = np.zeros((len(contour[0]), 3))
        points[:, 0:2] = contour[0] * cube.pixel_size

        points[:, 2] = i * cube.slice_distance
        c = Contour(points.tolist(), cube)
        c.contour_closed = True  # TODO: Probably the last point is double here
        s.add_contour(c)

        v.add_slice(s)
    return v


def create_cylinder(cube, name, center, radius, depth):
    """
    Creates a new VOI which holds the contours rendering a cylinder along z

    :param Cube cube: A CTX or DOS cube to work on.
    :param str name: Name of the VOI
    :param [float*3] center: Center position of cylinder [x,y,z] in [mm]
    :param float radius: Radius of cylinder in [mm]
    :param float depth: Depth of cylinder, along z in [mm]
    :returns: A new Voi object.
    """
    v = Voi(name, cube)
    t = np.linspace(start=0, stop=2.0 * pi, num=99, endpoint=False)
    p = list(zip(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)))

    for i in range(0, cube.dimz):
        z = i * cube.slice_distance
        if center[2] - depth / 2 <= z <= center[2] + depth / 2:
            s = Slice(cube)
            s.thickness = cube.slice_distance
            points = [[x[0], x[1], z] for x in p]
            if points:
                c = Contour(points, cube)
                c.contour_closed = True
                s.add_contour(c)
                v.add_slice(s)
    return v


def create_sphere(cube, name, center, radius):
    """
    Creates a new VOI which holds the contours rendering a sphere along z

    :param Cube cube: A CTX or DOS cube to work on.
    :param str name: Name of the VOI
    :param [float*3] center: Center position of sphere [x,y,z] in [mm]
    :param float radius: Radius of sphere in [mm]
    :returns: A new Voi object.
    """
    v = Voi(name, cube)

    t = np.linspace(start=0, stop=2.0 * pi,
                    num=99, endpoint=False)  # num: sets the number of corners in sphere per slice.
    p = list(zip(np.cos(t), np.sin(t)))

    points = []

    for i in range(0, cube.dimz):
        z = i * cube.slice_distance
        if center[2] - radius <= z <= center[2] + radius:
            r2 = radius**2 - (z - center[2])**2
            s = Slice(cube)
            s.thickness = cube.slice_distance
            _contour_closed = True
            if r2 > 0.0:
                points = [[center[0] + x[0] * sqrt(r2),
                           center[1] + x[1] * sqrt(r2), z] for x in p]
            # in case r2 == 0.0, the contour in this slice is a point.
            # TODO: How should the sphere be treated with points in the end slices:
            # seen from the side: " .oOo. "  or should it be "  oOo  "  ?
            # The former means the voi consists of contours and points, which I am not sure is valid.
            # Here "  oOo  " is implemented.
            # If you do not want the " .oOo. " version uncomment the next three lines.
            else:
                points = [[center[0], center[1], z]]
            if len(points) > 0:
                c = Contour(points, cube)
                c.contour_closed = _contour_closed
                s.add_contour(c)
                v.add_slice(s)
    return v


class Voi:
    """
    This is a class for handling volume of interests (VOIs). This class can e.g. be found inside the VdxCube object.
    VOIs may for instance be organs (lung, eye...) or targets (PTV, GTV...), or any other volume of interest.
    """

    sagital = 2  #: deprecated, backwards compatibility to pytripgui, do not use.
    sagittal = 2  #: id for sagittal view
    coronal = 1  #: id for coronal view

    def __init__(self, name, cube=None):
        self.cube = cube
        self.name = name
        self.is_concated = False
        self.type = 90
        self.slices = []
        self.color = [0, 230, 0]  # default colour
        self.define_colors()

    def __str__(self):
        """ str output handler
        """
        return self._print()

    def _print(self):
        """ Pretty print current attributes.
        """
        out = "\n"
        out += "   Voi\n"
        out += "----------------------------------------------------------------------------\n"
        out += "|   Name                                : '{:s}'\n".format(self.name)
        out += "|   Is concated                         : {:s}\n".format(str(self.is_concated))
        out += "|   Type                                : {:d}\n".format(self.type)
        out += "|   Number of slices in VOI             : {:d}\n".format(len(self.slices))
        out += "|   Color 0xRGB                         : #{:s}{:s}{:s}\n".format(hex(self.color[0].strip('0x')),
                                                                                  hex(self.color[1].strip('0x')),
                                                                                  hex(self.color[2].strip('0x')))

        return out

    def create_copy(self, margin=0):
        """
        Returns an independent copy of the Voi object

        :param margin: (unused)
        :returns: a deep copy of the Voi object
        """
        voi = copy.deepcopy(self)
        if not margin == 0:
            pass
        return voi

    def get_voi_cube(self, level=1000, recalc=False):
        """
        This method returns a DosCube object with value 1000 in each voxel within the Voi and zeros elsewhere.
        It can be used as a mask, for selecting certain voxels.
        The function may take some time to execute the first invocation, but is faster for subsequent calls,
        due to caching.

        :param level: Level which will be set to every voxel which is inside VOI. 1000 by default.
        :param recalc: force recalculation (avoid caching)
        :returns: a DosCube object which holds the value <level> in those voxels which are inside the Voi.
        """
        # caching: checks if class has voi_cube attribute
        # this is needed for speedup.

        if not recalc and hasattr(self, "voi_cube"):
            _max = self.voi_cube.max()
            _max_inv = 1 / _max
            if _max == level:
                return self.voi_cube
            else:
                return self.voi_cube * _max_inv * level

        self.voi_cube = DosCube(self.cube)
        self.voi_cube.mask_by_voi_all(self, level)
        return self.voi_cube

    def add_slice(self, slice):
        """ Add another slice to this VOI, and update self.slice_z table.

        :param Slice slice: the Slice object to be appended.
        """
        self.slices.append(slice)

    def get_name(self):
        """
        :returns: The name of this VOI.
        """
        return self.name

    def calculate_bad_angles(self, voi):
        """
        (Not implemented.)
        """
        pass

    def concat_to_3d_polygon(self):
        """ Concats all contours into a single contour, and writes all data points to self.polygon3d.
        """
        self.concat_contour()
        data = []
        for sl in self.slices:
            data.extend(sl.contours[0].contour)
        self.polygon3d = np.array(data)

    def get_3d_polygon(self):
        """ Returns a list of points rendering a 3D polygon of this VOI, which is stored in
        sefl.polygon3d. If this attribute does not exist, create it.
        """
        if not hasattr(self, "polygon3d"):
            self.concat_to_3d_polygon()
        return self.polygon3d

    def create_point_tree(self):
        """
        Concats all contours.
        Writes a list of points into self.points describing this VOI.
        """
        points = {}
        self.concat_contour()

        for sl in self.slices:  # TODO: should be sorted
            contour = sl.contours[0].contour
            p = {}
            for x in contour:
                p[x[0], x[1], x[2]] = []
            points.update(p)
        n_slice = len(self.slices)
        last_contour = None

        for i, sl in enumerate(self.slices):
            contour = sl.contours[0].contour
            n_points = len(contour)
            if i < n_slice - 1:
                next_contour = self.slices[i + 1].contours[0].contour
            else:
                next_contour = None
            for j, point in enumerate(contour):
                j2 = (j + 1) % (n_points - 2)
                point2 = contour[j2]
                points[(point[0], point[1], point[2])].append(point2)
                points[(point2[0], point2[1], point2[2])].append(point)
                if next_contour is not None:
                    point3 = pytrip.res.point.get_nearest_point(point, next_contour)
                    points[(point[0], point[1], point[2])].append(point3)
                    points[(point3[0], point3[1], point3[2])].append(point)
                if last_contour is not None:
                    point4 = pytrip.res.point.get_nearest_point(point, last_contour)
                    if point4 not in points[(point[0], point[1], point[2])]:
                        points[(point[0], point[1], point[2])].append(point4)
                        points[(point4[0], point4[1], point4[2])].append(point)
            last_contour = contour
        self.points = points

    def get_2d_projection_on_basis(self, basis, offset=None):
        """
        (TODO: Documentation)
        """
        a = np.array(basis[0])
        b = np.array(basis[1])
        self.concat_contour()
        bas = np.array([a, b])
        data = self.get_3d_polygon()
        product = np.dot(data, np.transpose(bas))

        compare = self.cube.pixel_size
        filtered = pytriplib.filter_points(product, compare / 2.0)
        filtered = np.array(sorted(filtered, key=cmp_to_key(_voi_point_cmp)))
        filtered = pytriplib.points_to_contour(filtered)
        product = filtered

        if offset is not None:
            offset_proj = np.array([np.dot(offset, a), np.dot(offset, b)])
            product = product[:] - offset_proj
        return product

    def get_2d_slice(self, plane, depth):
        """
        Gets a 2d Slice object from the contour in either sagittal or coronal plane.
        Contours will be concatenated.

        :param int plane: either self.sagittal or self.coronal
        :param float depth: position of plane
        :returns: a Slice object.
        """
        self.concat_contour()
        points1 = []
        points2 = []
        for _slice in self.slices:  # TODO: slices must be sorted first, but wouldnt they always be ?
            if plane is self.sagittal:
                point = sorted(
                    pytriplib.slice_on_plane(np.array(_slice.contours[0].contour), plane, depth), key=lambda x: x[1])
            elif plane is self.coronal:
                point = sorted(
                    pytriplib.slice_on_plane(np.array(_slice.contours[0].contour), plane, depth), key=lambda x: x[0])
            if len(point) > 0:
                points2.append(point[-1])
                if len(point) > 1:
                    points1.append(point[0])
        s = None
        if len(points1) > 0:
            points1.extend(reversed(points2))
            points1.append(points1[0])
            s = Slice(cube=self.cube)
            s.add_contour(Contour(points1, cube=self.cube))

        return s

    def define_colors(self):
        """
        Creates a list of default colours [R,G,B] in self.colours.
        """
        self.colors = []
        self.colors.append([0, 0, 255])
        self.colors.append([0, 128, 0])
        self.colors.append([0, 255, 0])
        self.colors.append([255, 0, 0])
        self.colors.append([0, 128, 128])
        self.colors.append([255, 255, 0])

    def calculate_center(self):
        """
        Calculates the center of gravity for the VOI.

        :returns: A numpy array[x,y,z] with positions in [mm]
        """
        if hasattr(self, "center_pos"):
            return self.center_pos
        self.concat_contour()
        tot_volume = 0.0
        center_pos = np.array([0.0, 0.0, 0.0])
        for _slice in self.slices:
            center, area = _slice.calculate_center()
            tot_volume += area
            center_pos += area * center
        if tot_volume > 0:
            self.center_pos = center_pos / tot_volume
            return center_pos / tot_volume
        else:
            self.center_pos = center_pos
            return center_pos

    def get_color(self, i=None):
        """
        :param int i: selects a colour, default if None.
        :returns: a [R,G,B] list.
        """
        if i is None:
            return self.color
        return self.colors[i % len(self.colors)]

    def set_color(self, color):
        """
        :param [3*int]: set a color [R,G,B].
        """
        self.color = color

    def create_dicom_label(self):
        """ Based on self.name and self.type, a Dicom ROI_LABEL is generated.

        :returns: a Dicom ROI_LABEL
        """
        roi_label = Dataset()
        roi_label.ROIObservationLabel = self.name
        roi_label.RTROIInterpretedType = self.get_roi_type_name(self.type)
        return roi_label

    def create_dicom_structure_roi(self):
        """ Based on self.name, an empty Dicom ROI is generated.

        :returns: a Dicom ROI.
        """
        roi = Dataset()
        roi.ROIName = self.name
        return roi

    def create_dicom_contour_data(self, i):
        """
        Based on self.slices, DICOM contours are generated for the DICOM ROI.

        :returns: Dicom ROI_CONTOURS
        """
        roi_contours = Dataset()
        contours = []

        dcmcube = None
        if self.cube is not None:
            dcmcube = self.cube.create_dicom()

        for _slice in self.slices:
            logger.info("Get contours from slice at {:10.3f} mm".format(_slice.get_position()))
            contours.extend(_slice.create_dicom_contours(dcmcube))

        roi_contours.Contours = Sequence(contours)
        roi_contours.ROIDisplayColor = self.get_color(i)

        return roi_contours

    def _sort_slices(self):
        """
        Sorts all slices stored in self for increasing z.
        This is needed for displaying Saggital and Coronal view.
        """
        # slice_in_frame is only given by VDX, and these are also the only frames which need to be sorted
        # it seems. DICOM apparently have proper structure already. Nonetheless, this function is also
        # applied to DICOM contours.

        if len(self.slices) > 0 and hasattr(self.slices[0], "slice_in_frame"):
            self.slices.sort(key=lambda _slice: _slice.slice_in_frame, reverse=True)

    def read_vdx_old(self, content, i):
        """
        Reads a single VOI from Voxelplan .vdx data from 'content', assuming a legacy .vdx format.
        VDX format 1.2.

        :params [str] content: list of lines with the .vdx content
        :params int i: line number to the list.
        :returns: current line number, after parsing the VOI.
        """

        logger.debug("Reading legacy 1.2 VDX format.")
        line = content[i]
        items = line.split()
        self.name = items[1]
        self.type = int(items[3])
        i += 1
        while i < len(content):
            line = content[i].strip()
            if line.startswith("voi"):
                break
            if line.startswith("slice#"):
                s = Slice(cube=self.cube)
                i = s.read_vdx_old(content, i)  # Slices in .vdx files start at 0
                if self.cube is not None:
                    for cont1 in s.contours:
                        for cont2 in cont1.contour:
                            _slice_number = int(cont2[2])
                            # bound checking
                            if _slice_number > self.cube.dimz:
                                logger.error("VDX slice number# {:d} exceeds dimension of CTX "
                                             "cube zmax={:d}".format(_slice_number, self.cube.dimz))
                                raise Exception("VDX file not compatible with CTX cube")

                            # cont2[2] holds slice number (starting in 1), translate it to absolute position in [mm]
                            cont2[2] = self.cube.slice_to_z(int(cont2[2]))
                if s.get_position() is None:
                    raise Exception("cannot calculate slice position")

                self.slices.append(s)

            if line.startswith("#TransversalObjects"):
                pass
                # slices = int(line.split()[1]) # TODO holds information about number of skipped slices
            i += 1

        self._sort_slices()
        return i - 1

    def read_vdx(self, content, i):
        """
        Reads a single VOI from Voxelplan .vdx data from 'content'.
        VDX format version 2.0

        :params [str] content: list of lines with the .vdx content
        :params int i: line number to the list.
        :returns: current line number, after parsing the VOI.
        """
        logger.debug("Parsing VDX format 2.0")

        line = content[i]
        self.name = ' '.join(line.split()[1:])
        number_of_slices = 10000
        i += 1
        while i < len(content):
            line = content[i].strip()
            if line.startswith("key"):
                self.key = line.split()[1]
            elif line.startswith("type"):
                self.type = int(line.split()[1])
            elif line.startswith("number_of_slices"):
                number_of_slices = int(line.split()[1])
            elif line.startswith("slice "):  # need that extra space to discriminate from "slice_in_frame"
                s = Slice(cube=self.cube)
                i = s.read_vdx(content, i)
                if s.get_position() is None:
                    raise Exception("cannot calculate slice position")
                if self.cube is None:
                    raise Exception("cube not loaded")
                self.slices.append(s)

            elif line.startswith("voi"):
                break
            elif len(self.slices) >= number_of_slices:
                break
            i += 1

        self._sort_slices()
        return i - 1

    def get_roi_type_number(self, type_name):
        """
        :returns: 1 if GTV or CTV, 10 for EXTERNAL, else 0.
        """
        if type_name == 'EXTERNAL':
            return 10
        elif type_name == 'AVOIDANCE':
            return 2
        elif type_name == 'ORGAN':
            return 0
        elif type_name == 'GTV':
            return 1
        elif type_name == 'CTV':
            return 1
        else:
            return 0

    def get_roi_type_name(self, type_id):
        """
        :returns: The type name of the ROI.
        """
        if type_id == 10:
            return "EXTERNAL"
        elif type_id == 2:
            return 'AVOIDANCE'
        elif type_id == 1:
            return 'CTV'
        elif type_id == 0:
            return 'ORGAN'
        return ''

    def read_dicom(self, roi_obs, roi_cont, roi_name="(none)"):
        """
        Reads a single ROI (= VOI) from a Dicom data set.

        :param roi_obs: dcm['rtss'].RTROIObservationsSequence[i]
        :param roi_cont: dcm['rtss'].ROIContourSequence[i]
        :param roi_name: the name of this ROI.
        """

        if not hasattr(roi_cont, "ContourSequence"):
            logger.warning("No DICOM (3006,0050) Contour Data found in "
                           "(3006,0039) ROIContourSequence[] for ROI:'{}'".format(roi_name))
            return

        _roi_type_name = roi_obs.RTROIInterpretedType
        self.type = self.get_roi_type_number(_roi_type_name)
        self.color = roi_cont.ROIDisplayColor

        contours = roi_cont.ContourSequence
        for i, contour in enumerate(contours):

            # get current slice position
            _z_pos = contour.ContourData[2]

            # if we have a new z_position, add a new slice object to self.slices
            if _z_pos not in [_slice.get_position() for _slice in self.slices]:
                logger.debug("VOI {}: Append new slice at z = {:f} cm to slices list:".format(self.name, _z_pos))
                sl = Slice(cube=self.cube)
                self.slices.append(sl)
            else:
                # lookup proper slice (just to be sure, should the contours come in random order)
                logger.debug("VOI {}: Found multi-contour at z = {} cm".format(self.name, _z_pos))
                sl = self.get_slice_at_pos(_z_pos)

            # append the contour data to the contour list of this slice
            sl.add_dicom_contour(contour)

            # 'CLOSED_PLANAR' indicates that the last point shall be connected to the first point,
            # where the first point is not repeated in the Contour Data.
            #
            # 'POINT' indicates that the contour is a single point, defining a specific location of significance.
            #
            # 'OPEN_PLANAR' indicates that the last vertex shall not be connected to the first point,
            # and that all points in Contour Data (3006,0050) shall be coplanar.
            #
            # 'OPEN_NONPLANAR' indicates that the last vertex shall not be connected to the first point,
            # and that the points in Contour Data (3006,0050) may be non-coplanar.
            # This can be used to represent objects best described by a single, possibly non-coplanar curve,
            # such as a brachytherapy applicator.
            #
            # Reference: https://www.dabsoft.ch/dicom/3/C.8.8.6.1/

            # Apparently DICOM may have single points as contours which can be marked as either POINT or CLOSED_PLANAR.
            # Here, we will let any contour which is less than 3 points be an open contour per definition.
            # example a CLOSED_PLANAR point:
            # SLICERRT: pinnacle3-9.9-phantom-imrt
            # - ROIContourSequence[2].ContourSequence[16].ContourGeometricType

            if contour.NumberOfContourPoints > 2 and contour.ContourGeometricType == 'CLOSED_PLANAR':
                sl.contours[-1].contour_closed = True
            else:
                sl.contours[-1].contour_closed = False

        self._sort_slices()

    def vdx_string(self):
        """
        Returns list of strings for this voi in voxelplan format, which can be written into a .vdx file.
        VDX format 2.0 only.

        :returns: a list of str holding the all lines needed for a Voxelplan formatted file.
        """
        if len(self.slices) is 0:
            return ""

        out = "\n"
        out += "voi %s\n" % (self.name.replace(" ", "_"))
        out += "key empty\n"
        out += "type %s\n" % self.type
        out += "\n"
        out += "contours\n"
        out += "reference_frame\n"
        out += " origin 0.000 0.000 0.000\n"
        out += " point_on_x_axis 1.000 0.000 0.000\n"
        out += " point_on_y_axis 0.000 1.000 0.000\n"
        out += " point_on_z_axis 0.000 0.000 1.000\n"
        out += "number_of_slices %d\n" % self.number_of_slices()
        out += "\n"
        i = 0

        for sl in self.slices:
            pos = sl.get_position()
            out += "slice {:d}\n".format(i)
            out += "slice_in_frame {:.3f}\n".format(pos)
            out += "thickness {:.3f} reference start_pos {:.3f} stop_pos {:.3f}\n".format(sl.thickness,
                                                                                          pos - 0.5 * sl.thickness,
                                                                                          pos + 0.5 * sl.thickness)
            out += "number_of_contours {:d}\n".format(sl.number_of_contours())
            out += sl.vdx_string()
            i += 1
        return out

    def get_row_intersections(self, pos):
        """
        For a given postion pos, returns a sorted list of x-coordinates where the y-coordinate intersects the contours.

        :param pos: a 3D position in the form [x,y,z] (mm)
        :returns: a sorted list of all x coordinates of all contours intersecting with the contours.

        TODO: could be made static
        TODO: could be made private
        """
        _slice = self.get_slice_at_pos(pos[2])
        if _slice is None:
            return None
        return np.sort(_slice.get_intersections(pos))

    def get_slice_at_pos(self, z):
        """
        Finds and returns a slice object found at position z [mm] (float).

        :param float z: slice position in absolute coordiantes (i.e. including any offsets)
        :returns: VOI slice at position z, z position may be approxiamte
        """

        _slice = [item for item in self.slices if np.isclose(item.get_position(), z,
                                                             atol=item.thickness * 0.5)]
        if len(_slice) == 0:
            logger.debug("could not find slice in get_slice_at_pos() at position {}".format(z))
            return None
        else:
            logger.debug("found slice at pos for z: {:.2f} mm, thickness {:.2f} mm".format(z, _slice[0].thickness))
            return _slice[0]

    def number_of_slices(self):
        """
        :returns: number of slices covered by this VOI.
        """
        return len(self.slices)

    def concat_contour(self):
        """ Concat all contours in all slices found in this VOI.
        """
        if not self.is_concated:
            for sl in self.slices:
                sl.concat_contour()
        self.is_concated = True

    def get_min_max(self):
        """ Set self.temp_min and self.temp_max if they dont exist.

        :returns: minimum and maximum x y coordinates in Voi.
        """
        temp_min, temp_max = None, None
        if hasattr(self, "temp_min"):
            return self.temp_min, self.temp_max
        for _slice in self.slices:
            if temp_min is None:
                temp_min, temp_max = _slice.get_min_max()
            else:
                min1, max1 = _slice.get_min_max()
                temp_min = pytrip.res.point.min_list(temp_min, min1)
                temp_max = pytrip.res.point.max_list(temp_max, max1)
        self.temp_min = temp_min
        self.temp_max = temp_max
        return temp_min, temp_max


class Slice:
    """
    The Slice class is specific for structures, and should not be confused with Slices extracted from CTX or DOS
    objects.
    """

    def __init__(self, cube=None):
        self.cube = cube
        self.contours = []  # list of contours in this slice

        # the slice positions are recorded, however the thickness
        # may be smaller than just the distance between two slices.
        # Therefore it is here set to some small non-zero value.
        self.thickness = 0.1  # assume some default value
        if cube is not None:
            self.thickness = cube.slice_distance

    def add_contour(self, contour):
        """ Adds a new 'contour' to the existing contours.

        :param Contour contour: the contour to be added.
        """
        self.contours.append(contour)

    def add_dicom_contour(self, dcm):
        """ Adds a Dicom CONTOUR to the existing list of contours in this Slice class.

        :param Dicom dcm: a Dicom CONTOUR object.
        """

        # do not apply any offset here, since everything is written in real world coordinates.
        _offset = [0.0, 0.0, 0.0]
        self.contours.append(
            Contour(pytrip.res.point.array_to_point_array(np.array(dcm.ContourData, dtype=float), _offset),
                    self.cube))
        # add the slice position to slice_in_frame which is needed later for sorting.
        self.slice_in_frame = self.contours[-1].contour[0][2]

    def get_position(self):
        """
        :returns: the position of this slice in [mm] including zoffset
        """
        if len(self.contours) == 0:
            return None
        return self.contours[0].contour[0][2]

    def get_intersections(self, pos):
        """
        For a given position <pos>, return a list of x-coordinates intersecting where the pos.y intersects all
        closed contours found in this slice.

        :params pos: a position in the form [x,y]
        :returns: a list of x-coordinates intersecting the y-coordinate found in pos[1].

        :warning: method name may change.
        TODO: rename this method. It could possibly be merged with get_x_intersections.
        TODO: could be made static
        TODO: could be made private
        """
        intersections = []
        for c in self.contours:
            # do not include open contours.
            if c.contour_closed:
                intersections.extend(pytrip.res.point.get_x_intersection(pos[1], c.contour))
        return intersections

    def calculate_center(self):
        """ Calculate the center position of all contours in this slice.

        :returns: a list of center positions [x,y,z] in [mm] for each contour found.
        """
        tot_area = 0.0
        center_pos = np.array([0.0, 0.0, 0.0])
        for contour in self.contours:
            center, area = contour.calculate_center()
            tot_area += area
            center_pos += area * center
        if tot_area > 0:
            return center_pos / tot_area, tot_area
        else:
            return center_pos, tot_area

    def read_vdx(self, content, i):
        """
        Reads a single Slice from Voxelplan .vdx data from 'content'.
        VDX format 2.0.

        :params [str] content: list of lines with the .vdx content
        :params int i: line number to the list.
        :returns: current line number, after parsing the VOI.
        """
        self.thickness = 0.1  # some small default value
        line = content[i]
        number_of_contours = 0
        i += 1
        while i < len(content):
            line = content[i].strip()
            if line.startswith("slice_in_frame"):
                self.slice_in_frame = float(line.split()[1])
            elif line.startswith("thickness"):
                items = line.split()
                self.thickness = float(items[1])
                logger.debug("Read VDX: thickness = {:f}".format(self.thickness))

                if len(items) == 7:
                    self.start_pos = float(items[4])
                    self.stop_pos = float(items[6])
                else:
                    self.start_pos = float(items[3])
                    self.stop_pos = float(items[5])

            elif line.startswith("number_of_contours"):
                number_of_contours = int(line.split()[1])
            elif line.startswith("contour"):
                c = Contour([], self.cube)
                i = c.read_vdx(content, i)
                self.add_contour(c)
                # TODO: not sure if multiple contours for the same ROI/VOI are allowed in VDX format.
            elif line.startswith("slice "):  # need that extra space to discriminate from "slice_in_frame"
                break
            elif len(self.contours) >= number_of_contours:
                break
            i += 1
        return i - 1

    def read_vdx_old(self, content, i):
        """
        Reads a single Slice from Voxelplan .vdx data from 'content'.
        VDX format 1.2.

        :params [str] content: list of lines with the .vdx content
        :params int i: line number to the list.
        :returns: current line number, after parsing the VOI.
        """

        # VDX cubes in version 1.2 do not hold any information on slice thicknesses.
        line1 = content[i].strip()
        line2 = content[i + 1].strip()
        line3 = content[i + 2].strip()

        if not line1.startswith("slice#"):
            return None
        if not line2.startswith("#points"):
            return None
        if not line3.startswith("points"):
            return None

        self.slice_in_frame = float(line1.split()[1])

        c = Contour([], self.cube)
        c.read_vdx_old(slice_number=self.slice_in_frame, xy_line=line3.split()[1:])
        self.add_contour(c)

        return i

    def create_dicom_contours(self, dcmcube):
        """
        Creates and returns a list of Dicom CONTOUR objects from self.

        :param dcmcube: TODO write me
        """

        # in order to get DICOM readable by Eclipse we need to connect each contour with CT slice
        # CT slices are identified by SOPInstanceUID
        # first we assume some default value if we cannot figure out CT slice info (i.e. CT cube is not loaded)
        ref_sop_instance_uid = '1.2.3'

        # then we check if CT cube is loaded
        if dcmcube is not None:
            candidates = [dcm for dcm in dcmcube if np.isclose(dcm.SliceLocation,
                                                               self.get_position())]
            if len(candidates) > 0:
                # finally we extract CT slice SOP Instance UID
                ref_sop_instance_uid = candidates[0].SOPInstanceUID

        contour_list = []
        for item in self.contours:
            con = Dataset()
            contour = []
            for p in item.contour:
                contour.extend([p[0], p[1], p[2]])
            con.ContourData = contour
            con.ContourGeometricType = 'CLOSED_PLANAR'
            con.NumberofContourPoints = item.number_of_points()
            cont_image_item = Dataset()
            cont_image_item.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage SOP Class
            cont_image_item.ReferencedSOPInstanceUID = ref_sop_instance_uid  # CT slice Instance UID
            con.ContourImageSequence = Sequence([cont_image_item])
            contour_list.append(con)
        return contour_list

    def vdx_string(self):
        """
        Returns list of strings for this SLICE in voxelplan format, which can be written into a .vdx file.
        VDX format 2.0 only.

        :returns: a list of str holding the slice information with the countour lines for a Voxelplan formatted file.
        """
        out = ""
        for i, cnt in enumerate(self.contours):
            out += "contour %d\n" % i
            out += "internal false\n"

            if cnt.number_of_points() == 1:  # Handle POIs
                out += "number_of_points {:d}\n".format(cnt.number_of_points())
            else:  # Handle ROIs
                out += "number_of_points {:d}\n".format(cnt.number_of_points() + 1)
            out += cnt.vdx_string()
            out += "\n"
        return out

    def number_of_contours(self):
        """
        :returns: number of contours found in this Slice object.
        """
        return len(self.contours)

    def concat_contour(self):
        """
        Concat all contours in this Slice object to a single contour.
        """
        for i in range(len(self.contours) - 1, 0, -1):
            self.contours[0].push(self.contours[i])
            self.contours.pop(i)
        self.contours[0].concat()

    def remove_inner_contours(self):
        """
        Removes any "holes" in the contours of this slice, thereby changing the topology of the contour.
        """
        for i in range(len(self.contours) - 1, 0, -1):
            self.contours[0].push(self.contours[i])
            self.contours.pop(i)
        self.contours[0].remove_inner_contours()

    def get_min_max(self):
        """
        Set self.temp_min and self.temp_max if they dont exist.

        :returns: minimum and maximum x y coordinates in Voi.
        """
        temp_min, temp_max = self.contours[0].get_min_max()
        for i in range(1, len(self.contours)):
            min1, max1 = self.contours[i].get_min_max()
            temp_min = pytrip.res.point.min_list(temp_min, min1)
            temp_max = pytrip.res.point.max_list(temp_max, max1)
        return temp_min, temp_max


class Contour:
    """
    Class for handling single Contours.
    The contour class holds a list of points self.contour = [[x0,y0,z0], [x1,y1,z1], ... [xn, yn, zn]] in millimeters.
    A contour can also be a single point (POI).
    A contour may be open or closed.
    """

    def __init__(self, contour, cube=None):
        self.cube = cube
        self.children = []
        self.contour = contour  # TODO: consider renaming this to 'data' or 'contour_data'
        # contour_closed: if set to True, the last point in the contour will be repeated when writing VDX files.
        self.contour_closed = False

    def push(self, contour):
        """
        Push a contour on the contour stack.

        :param Contour contour: a Contour object.
        """
        for i in range(len(self.children)):
            if self.children[i].contains_contour(contour):
                self.children[i].push(contour)
                return
        self.add_child(contour)

    def calculate_center(self):
        """
        Calculate the center for a single contour, and the area of a contour in 3 dimensions.

        :returns: Center of the contour [x,y,z] in [mm], area [mm**2] (TODO: to be confirmed)
        """
        points = np.array(self.contour + [self.contour[0]])  # connect the contour first and last point
        # its needed only for calculation of dxdy in Green's theorem below
        dx_dy = np.diff(points, axis=0)
        if abs(points[0, 2] - points[1, 2]) < 0.01:
            area = -np.dot(points[:-1, 1], dx_dy[:, 0])
            paths = (dx_dy[:, 0]**2 + dx_dy[:, 1]**2)**0.5
        elif abs(points[0, 1] - points[1, 1]) < 0.01:
            area = -np.dot(points[:-1, 2], dx_dy[:, 0])
            paths = (dx_dy[:, 0]**2 + dx_dy[:, 2]**2)**0.5
        elif abs(points[0, 0] - points[1, 0]) < 0.01:
            area = -np.dot(points[:-1, 2], dx_dy[:, 1])
            paths = (dx_dy[:, 1]**2 + dx_dy[:, 2]**2)**0.5
        total_path = np.sum(paths)

        if total_path > 0:
            center = np.array([np.dot(points[:-1, 0], paths) / total_path,
                               np.dot(points[:-1, 1], paths) / total_path,
                               points[0, 2]])
        else:
            center = np.array([np.sum(points[:-1, 0]),
                               np.sum(points[:-1, 1]),
                               points[0, 2]])

        return center, area

    def get_min_max(self):
        """
        :returns: The lowest x,y,z values and the highest x,y,z values found in this Contour object.
        """
        min_x = np.amin(np.array(self.contour)[:, 0])
        min_y = np.amin(np.array(self.contour)[:, 1])
        min_z = np.amin(np.array(self.contour)[:, 2])

        max_x = np.amax(np.array(self.contour)[:, 0])
        max_y = np.amax(np.array(self.contour)[:, 1])
        max_z = np.amax(np.array(self.contour)[:, 2])
        return [min_x, min_y, min_z], [max_x, max_y, max_z]

    def vdx_string(self):
        """
        Returns list of strings for this CONTOUR in voxelplan format, which can be written into a .vdx file.
        VDX format 2.0 only.

        :returns: an array of str holding the contour points needed for a Voxelplan formatted file.
        """

        # The vdx files require all contours to be mapped to a CT cube starting at (x,y) = (0,0)
        # However, z may be mapped directly as found in the dicom file by using z_tables in .hed
        out = ""
        for i, cnt in enumerate(self.contour):
            out += " %.4f %.4f %.4f %.4f %.4f %.4f\n" % (cnt[0] - self.cube.xoffset,
                                                         cnt[1] - self.cube.yoffset,
                                                         cnt[2],
                                                         0, 0, 0)

        # repeat the first point, to close the contour, if needed
        if self.contour_closed and len(self.contour) > 1:
            out += " %.4f %.4f %.4f %.4f %.4f %.4f\n" % (self.contour[0][0] - self.cube.xoffset,
                                                         self.contour[0][1] - self.cube.yoffset,
                                                         self.contour[0][2],
                                                         0, 0, 0)
        return out

    def read_vdx(self, content, i):
        """
        Reads a single Contour from Voxelplan .vdx data from 'content'.
        VDX format 2.0.

       Note:
        - in VDX files the last contour point is repeated if the contour is closed.
        - If we have a point of interest, the length is 1.
        - Length 2 and 3 should thus never occur in VDX files (assuming all contours are closed)

        :params [str] content: list of lines with the .vdx content
        :params int i: line number to the list.
        :returns: current line number, after parsing the VOI.
        """
        self.contour_closed = False
        set_point = False
        points = 0
        j = 0
        while i < len(content):
            line = content[i].strip()

            # skip any empty lines
            if line == "":
                i += 1  # go to next line
                continue

            if set_point:  # expect list of x,y,z points
                if j >= points:
                    break

                con_dat = line.split()
                if len(con_dat) < 3:  # point must be at least three dimensional
                    logger.warning(".vdx line {:d}: ignored, expected <x> <y> <z> positions".format(i))
                else:
                    self.contour.append([float(con_dat[0]) + self.cube.xoffset,
                                         float(con_dat[1]) + self.cube.yoffset,
                                         float(con_dat[2])])
                    j += 1  # increment point counter

            else:  # in case we do not have a point, some keyword may be found
                if line.startswith("internal_false"):
                    self.internal_false = True
                if line.startswith("number_of_points"):
                    points = int(line.split()[1])
                    set_point = True
            i += 1  # go to next line

        # check if the contour is closed
        # self.contour[:] holds the actual data points
        if len(self.contour) > 1:  # check if this is an actual contour, and not a POI
            # if first data point is the same as the last data point we have closed contour
            if self.contour[0] == self.contour[-1]:
                self.contour_closed = True
                # and trash the last data point
                del self.contour[-1]
            else:
                self.contour_closed = False

        return i - 1

    def read_vdx_old(self, slice_number, xy_line):
        """
        Reads a single Contour from Voxelplan .vdx data from 'content' and appends it to self.contour data
        VDX format 1.2.

        See also notes in read_vdx(), regarding the length of a contour.

        :params slice_number: list of numbers (as characters) with slice number
        :params xy_line: list of numbers (as characters) representing X and Y coordinates of a contour
        """
        self.contour_closed = False

        if self.cube is None:
            _pixel_size = 1.0
            _warn = "Reading contour data in .vdx in 1.2 format: no CTX cube associated, setting pixel_size to 1.0."
            logger.warning(_warn)
        else:
            _pixel_size = self.cube.pixel_size

        # and example of xy_line: 3021 4761 2994 4899 2916 5015
        xy_pairs = [xy_line[i:i + 2] for i in range(0, len(xy_line), 2)]  # make list of pairs
        for x, y in xy_pairs:
            # TRiP98 saves X,Y coordinates as integers, to get [mm] they needs to be divided by 16
            self.contour.append([_pixel_size * float(x) / 16.0,
                                 _pixel_size * float(y) / 16.0,
                                 float(slice_number)])

        # The legacy 1.2 VDX format does not discriminate between open or closed contours.
        # Therefore all contours read will be closed, except for POIs.
        # I checked VDX 1.2 files generated by TRiP and converted from dcm2trip, the last point
        # is not repeated here.
        if len(self.contour) > 1:  # check if this is an actual contour, and not a POI
            logging.debug("VDX 1.2 read, contour is closed")
            self.contour_closed = True
        else:
            logging.debug("VDX 1.2 read, contour is a POI")
            self.contour_closed = False

    def add_child(self, contour):
        """ (TODO: Document me)
        """
        remove_idx = []
        for i, child in enumerate(self.children):
            if contour.contains_contour(child):
                contour.push(child)
                remove_idx.append(i)
        remove_idx.sort(reverse=True)
        for i in remove_idx:
            self.children.pop(i)
        self.children.append(contour)

    def number_of_points(self):
        """
        :returns: Number of points in this Contour object.
        """
        return len(self.contour)

    def has_childs(self):
        """
        :returns: True or False, whether this Contour object has children.
        """
        if len(self.children) > 0:
            return True
        return False

    def print_child(self, level):
        """
        Print child to stdout.

        :param int level: (TODO: needs documentation)
        """
        for i, item in enumerate(self.children):
            print(level * '\t', )
            print(item.contour)
            self.item.print_child(level + 1)

    def contains_contour(self, contour):
        """
        :returns: True if contour in argument is contained inside self.
        """
        return pytrip.res.point.point_in_polygon(contour.contour[0][0], contour.contour[0][1], self.contour)

    def concat(self):
        """
        In case of multiple contours in the same slice, this method will concat them to a single contour.
        This is important for TRiP98 compatibility, as TRiP98 cannot handle multiple contours in the same slice of
        of the same VOI.
        """
        for i in range(len(self.children)):
            self.children[i].concat()
        while len(self.children) > 1:
            d = -1
            child = 0
            for i in range(1, len(self.children)):
                i1_temp, i2_temp, d_temp = pytrip.res.point.short_distance_polygon_idx(
                    self.children[0].contour, self.children[i].contour)
                if d == -1 or d_temp < d:
                    d = d_temp
                    child = i
            i1_temp, i2_temp, d_temp = pytrip.res.point.short_distance_polygon_idx(
                self.children[0].contour, self.contour)
            if d_temp < d:
                self._merge(self.children[0])
                self.children.pop(0)
            else:
                self.children[0]._merge(self.children[child])
                self.children.pop(child)
        if len(self.children) == 1:
            self._merge(self.children[0])
            self.children.pop(0)

    def remove_inner_contours(self):
        """ (TODO: needs documentation)
        """
        for i in range(len(self.children)):
            self.children[i].children = []

    def _merge(self, contour):
        """
        Merge two contours into a single one.
        """
        if len(self.contour) == 0:
            self.contour = contour.contour
            return
        i1, i2, d = pytrip.res.point.short_distance_polygon_idx(self.contour, contour.contour)
        con = []
        for i in range(i1 + 1):
            con.append(self.contour[i])
        for i in range(i2, len(contour.contour)):
            con.append(contour.contour[i])
        for i in range(i2 + 1):
            con.append(contour.contour[i])
        for i in range(i1, len(self.contour)):
            con.append(self.contour[i])
        self.contour = con
        return
