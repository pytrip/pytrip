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
This module holds all the user needs to deal with Volume of interests.
It provides the top-level VdxCube class, Voi, Slice and Contour classes.
The Voi class represents a volume of interest 'VOI', also called region of interest 'ROI' in Dicom lingo.
Each Voi holds several Slice, which are noramlly synced with an associated CT-cube.
Each Slice holds one or more Contours.
"""
import os
import re
import copy
from math import pi
import logging
from functools import cmp_to_key

import numpy as np

import pytrip
from pytrip.error import InputError, ModuleNotLoadedError
from pytrip.dos import DosCube
import pytriplib

try:
    from dicom.dataset import Dataset, FileDataset
    from dicom.sequence import Sequence
    from dicom import UID

    _dicom_loaded = True
except:
    _dicom_loaded = False

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
    entity using infinte thin connects.

    VdxCube can import both dicom data and TRiP data,
    and export in the those formats.

    We strongly recommend to load a CT and/or a DOS cube first, see example below:

    >>> c = CtxCube()
    >>> c.read("TST000000")
    >>> v = VdxCube("", c)
    >>> v.read("TST000000.vdx")
    """
    def __init__(self, content, cube=None):
        self.vois = []
        self.cube = cube
        self.version = "1.2"

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
        for i, item in enumerate(dcm.ROIContours):
            if structure_ids is None or item.RefdROINumber in structure_ids:
                v = Voi(dcm.RTROIObservations[i].ROIObservationLabel.decode('iso-8859-1'), self.cube)
                v.read_dicom(dcm.RTROIObservations[i], item)
                self.add_voi(v)
        self.cube.xoffset = 0
        self.cube.yoffset = 0
        self.cube.zoffset = 0
        """shift = min(self.cube.slice_pos)
        for i in range(len(self.cube.slice_pos)):
                self.cube.slice_pos[i] = self.cube.slice_pos[i]-shift"""

    def get_voi_names(self):
        """
        :returns: a list of available voi names.
        """
        names = [voi.name for voi in self.vois]
        return names

    def __str__(self):  # Method for printing
        """
        :returns: VOI names separated by '&' sign
        """
        return '&'.join(self.get_voi_names())

    def add_voi(self, voi):
        """ Appends a new voi to this class.

        :param Voi voi: the voi to be appened to this class.
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

    def read_vdx(self, path):
        """ Reads a structure file in Voxelplan format.

        :param str path: Full path including file extension.
        """
        fp = open(path, "r")
        content = fp.read().split('\n')
        fp.close()
        i = 0
        n = len(content)
        header_full = False
        #        number_of_vois = 0
        while i < n:
            line = content[i]
            if not header_full:
                if re.match("vdx_file_version", line) is not None:
                    self.version = line.split()[1]
                elif re.match("all_indices_zero_based", line) is not None:
                    self.zero_based = True
#                TODO number_of_vois not used
#                elif re.match("number_of_vois", line) is not None:
#                    number_of_vois = int(line.split()[1])
            if re.match("voi", line) is not None:
                v = Voi(line.split()[1], self.cube)
                if self.version == "1.2":
                    if not line.split()[5] == '0':
                        i = v.read_vdx_old(content, i)
                else:
                    i = v.read_vdx(content, i)
                self.add_voi(v)
                header_full = True
            i += 1

    def concat_contour(self):
        """ Loop through all available VOIs and check whether any have mutiple contours in a slice.
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

    def write_to_voxel(self, path):
        """ Writes all VOIs in voxelplan format.

        :param str path: Full path, including file extension (.vdx).
        """
        fp = open(path, "w")
        fp.write("vdx_file_version %s\n" % self.version)
        fp.write("all_indices_zero_based\n")
        fp.write("number_of_vois %d\n" % self.number_of_vois())
        self.vois = sorted(self.vois, key=lambda voi: voi.type, reverse=True)
        for voi in self.vois:
            fp.write(voi.to_voxel_string())
        fp.close()

    def write_to_trip(self, path):
        """ Writes all VOIs in voxelplan format, while ensuring no slice holds more than one contour.
        Identical to write().

        :param str path: Full path, including file extension (.vdx).
        """
        self.concat_contour()
        self.write_to_voxel(path)

    def write(self, path):
        """ Writes all VOIs in voxelplan format, while ensuring no slice holds more than one contour.
        Identical to write_to_trip().

        :param str path: Full path, including file extension (.vdx).
        """
        self.write_to_trip(path)

    def create_dicom(self):
        """ Generates and returns Dicom RTSTRUCT object, which holds all VOIs.

        :returns: a Dicom RTSTRUCT object holding any VOIs.
        """
        if _dicom_loaded is False:
            raise ModuleNotLoadedError("Dicom")
        meta = Dataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage SOP Class
        # see https://github.com/darcymason/pydicom/blob/master/pydicom/_uid_dict.py
        meta.MediaStorageSOPInstanceUID = "1.2.3"
        meta.ImplementationClassUID = "1.2.3.4"
        meta.TransferSyntaxUID = UID.ImplicitVRLittleEndian  # Implicit VR Little Endian - Default Transfer Syntax
        ds = FileDataset("file", {}, file_meta=meta, preamble=b"\0" * 128)
        if self.cube is not None:
            ds.PatientsName = self.cube.patient_name
        else:
            ds.PatientsName = ""
        ds.PatientID = "123456"
        ds.PatientsSex = '0'
        ds.PatientsBirthDate = '19010101'
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.AccessionNumber = ''
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage SOP Class
        ds.SOPInstanceUID = '1.2.3'  # !!!!!!!!!!
        ds.StudyInstanceUID = '1.2.3'  # !!!!!!!!!!
        ds.SeriesInstanceUID = '1.2.3'  # !!!!!!!!!!
        ds.FrameofReferenceUID = '1.2.3'  # !!!!!!!!!
        ds.SeriesDate = '19010101'  # !!!!!!!!
        ds.ContentDate = '19010101'  # !!!!!!
        ds.StudyDate = '19010101'  # !!!!!!!
        ds.SeriesTime = '000000'  # !!!!!!!!!
        ds.StudyTime = '000000'  # !!!!!!!!!!
        ds.ContentTime = '000000'  # !!!!!!!!!
        ds.StructureSetLabel = 'pyTRiP plan'
        ds.StructureSetDate = '19010101'
        ds.StructureSetTime = '000000'
        ds.StructureSetName = 'ROI'
        ds.Modality = 'RTSTRUCT'
        roi_label_list = []
        roi_data_list = []
        roi_structure_roi_list = []

        # to get DICOM which can be loaded in Eclipse we need to store information about UIDs of all slices in CT
        # first we check if DICOM cube is loaded
        if self.cube is not None:
            rt_ref_series_data = Dataset()
            rt_ref_series_data.SeriesInstanceUID = '1.2.3.4.5'
            rt_ref_series_data.ContourImageSequence = Sequence([])

            # each CT slice corresponds to one DICOM file
            for slice_dicom in self.cube.create_dicom():
                slice_dataset = Dataset()
                slice_dataset.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage SOP Class
                slice_dataset.ReferencedSOPInstanceUID = slice_dicom.SOPInstanceUID  # most important - slice UID
                rt_ref_series_data.ContourImageSequence.append(slice_dataset)

            rt_ref_study_seq_data = Dataset()
            rt_ref_study_seq_data.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'  # Study Component Management Class
            rt_ref_study_seq_data.ReferencedSOPInstanceUID = '1.2.3.4.5'
            rt_ref_study_seq_data.RTReferencedSeriesSequence = Sequence([rt_ref_series_data])

            rt_ref_frame_study_data = Dataset()
            rt_ref_frame_study_data.RTReferencedStudySequence = Sequence([rt_ref_study_seq_data])
            rt_ref_frame_study_data.FrameOfReferenceUID = '1.2.3.4.5'
            ds.ReferencedFrameOfReferenceSequence = Sequence([rt_ref_frame_study_data])

        for i in range(self.number_of_vois()):
            roi_label = self.vois[i].create_dicom_label()
            roi_label.ObservationNumber = str(i + 1)
            roi_label.ReferencedROINumber = str(i + 1)
            roi_label.RefdROINumber = str(i + 1)
            roi_contours = self.vois[i].create_dicom_contour_data(i)
            roi_contours.RefdROINumber = str(i + 1)
            roi_contours.ReferencedROINumber = str(i + 1)

            roi_structure_roi = self.vois[i].create_dicom_structure_roi()
            roi_structure_roi.ROINumber = str(i + 1)

            roi_structure_roi_list.append(roi_structure_roi)
            roi_label_list.append(roi_label)
            roi_data_list.append(roi_contours)
        ds.RTROIObservations = Sequence(roi_label_list)
        ds.ROIContours = Sequence(roi_data_list)
        ds.StructureSetROIs = Sequence(roi_structure_roi_list)
        return ds

    def write_dicom(self, directory):
        """ Generates a Dicom RTSTRUCT object from self, and writes it to disk.

        :param str directory: Diretory where the rtss.dcm file will be saved.
        """
        dcm = self.create_dicom()
        dcm.save_as(os.path.join(directory, "rtss.dcm"))


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
            points = [
                [center[0] - width / 2, center[1] - height / 2, z], [center[0] + width / 2, center[1] - height / 2, z],
                [center[0] + width / 2, center[1] + height / 2, z], [center[0] - width / 2, center[1] + height / 2, z],
                [center[0] - width / 2, center[1] - height / 2, z]
            ]
            c = Contour(points, cube)
            s.add_contour(c)
            v.add_slice(s)
    return v


def create_voi_from_cube(cube, name, value=100):
    """
    Creates a new VOI which holds the contours following an isodose lines.

    :param Cube cube: A CTX or DOS cube to work on.
    :param str name: Name of the VOI
    :param int value: The isodose value from which the countour will be generated from.
    :returns: A new Voi object.
    """
    v = Voi(name, cube)
    # there are some cases when this script is run on systems without DISPLAY variable being set
    # in such case matplotlib backend has to be explicitly specified
    # we do it here and not in the top of the file, as inteleaving imports with code lines is discouraged
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib._cntr as cntr
    for i in range(cube.dimz):
        x, y = np.meshgrid(np.arange(len(cube.cube[0, 0])), np.arange(len(cube.cube[0])))
        isodose_obj = cntr.Cntr(x, y, cube.cube[i])
        contour = isodose_obj.trace(value)
        s = Slice(cube)
        if not len(contour):
            continue
        points = np.zeros((len(contour[0]), 3))
        points[:, 0:2] = contour[0] * cube.pixel_size

        points[:, 2] = i * cube.slice_distance
        c = Contour(points.tolist(), cube)
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
    t = np.linspace(start=0, stop=2.0 * pi, num=100)
    p = list(zip(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)))
    for i in range(0, cube.dimz):
        z = i * cube.slice_distance
        if center[2] - depth / 2 <= z <= center[2] + depth / 2:
            s = Slice(cube)
            points = [[x[0], x[1], z] for x in p]
            if points:
                c = Contour(points, cube)
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
    t = np.linspace(start=0, stop=2.0 * pi, num=100)
    p = list(zip(np.cos(t), np.sin(t)))
    for i in range(0, cube.dimz):
        z = i * cube.slice_distance
        if center[2] - radius <= z <= center[2] + radius:
            r = (radius**2 - (z - center[2])**2)**0.5
            s = Slice(cube)
            points = [[center[0] + r * x[0], center[1] + r * x[1], z] for x in p]
            if len(points) > 0:
                c = Contour(points, cube)
                s.add_contour(c)
                v.add_slice(s)
    return v


class Voi:
    """
    This is a class for handling volume of interests (VOIs). This class can e.g. be found inside the VdxCube object.
    VOIs may for instance be organs (lung, eye...) or targets (PTV, GTV...), or any other volume of interest.
    """

    sagital = 2  #: deprecated, backwards compability to pytripgui, do not use.
    sagittal = 2  #: id for sagittal view
    coronal = 1  #: id for coronal view

    def __init__(self, name, cube=None):
        self.cube = cube
        self.name = name
        self.is_concated = False
        self.type = 90
        self.slice_z = []
        self.slices = {}
        self.color = [0, 230, 0]  # default colour
        self.define_colors()

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

    def get_voi_cube(self):
        """
        This method returns a DosCube object with value 1000 in each voxel within the Voi and zeros elsewhere.
        It can be used as a mask, for selecting certain voxels.
        The function may take some time to execute the first invocation, but is faster for subsequent calls,
        due to caching.

        :returns: a DosCube object which holds the value 1000 in those voxels which are inside the Voi.
        """
        if hasattr(self, "voi_cube"):  # caching: checks if class has voi_cube attribute
            # TODO: add parameter as argument to this function. Note, this needs to be compatible with
            # caching the cube. E.g. the method might be called twice with different arguments.
            return self.voi_cube
        self.voi_cube = DosCube(self.cube)
        self.voi_cube.load_from_structure(self, 1000)
        return self.voi_cube

    def add_slice(self, slice):
        """ Add another slice to this VOI, and update self.slice_z table.

        :param Slice slice: the Slice object to be appended.
        """
        key = int(slice.get_position() * 100)
        self.slice_z.append(key)
        self.slices[key] = slice

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
        """ Concats all contours into a single contour, and writes all data points to sefl.polygon3d.
        """
        self.concat_contour()
        data = []
        for slice in self.slices:
            data.extend(self.slices[slice].contour[0].contour)
        self.polygon3d = np.array(data)

    def get_3d_polygon(self):
        """ Returns a list of points rendering a 3D polygon of this VOI, which is stored in
        sefl.polygon3d. If this attibute does not exist, create it.
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
        slice_keys = sorted(self.slices.keys())
        for key in slice_keys:
            contour = self.slices[key].contour[0].contour
            p = {}
            for x in contour:
                p[x[0], x[1], x[2]] = []
            points.update(p)
        n_slice = len(slice_keys)
        last_contour = None
        for i, key in enumerate(slice_keys):
            contour = self.slices[key].contour[0].contour
            n_points = len(contour)
            if i < n_slice - 1:
                next_contour = self.slices[slice_keys[i + 1]].contour[0].contour
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
        """ (TODO: Documentation)
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
        """ Gets a 2d Slice object from the contour in either sagittal or coronal plane.
        Contours will be concated.

        :param int plane: either self.sagittal or self.coronal
        :param float depth: position of plane
        :returns: a Slice object.
        """
        self.concat_contour()
        points1 = []
        points2 = []
        for key in sorted(self.slice_z):
            slice = self.slices[key]
            if plane is self.sagittal:
                point = sorted(
                    pytriplib.slice_on_plane(np.array(slice.contour[0].contour), plane, depth), key=lambda x: x[1])
            elif plane is self.coronal:
                point = sorted(
                    pytriplib.slice_on_plane(np.array(slice.contour[0].contour), plane, depth), key=lambda x: x[0])
            if len(point) > 0:
                points2.append(point[-1])
                if len(point) > 1:
                    points1.append(point[0])
        s = None
        if len(points1) > 0:
            points1.extend(reversed(points2))
            points1.append(points1[0])
            s = Slice(cube=self.cube)
            s.add_contour(Contour(points1))
        return s

    def define_colors(self):
        """ Creates a list of default colours [R,G,B] in self.colours.
        """
        self.colors = []
        self.colors.append([0, 0, 255])
        self.colors.append([0, 128, 0])
        self.colors.append([0, 255, 0])
        self.colors.append([255, 0, 0])
        self.colors.append([0, 128, 128])
        self.colors.append([255, 255, 0])

    def calculate_center(self):
        """ Calculates the center of gravity for the VOI.

        :returns: A numpy array[x,y,z] with positions in [mm]
        """
        if hasattr(self, "center_pos"):
            return self.center_pos
        self.concat_contour()
        tot_volume = 0.0
        center_pos = np.array([0.0, 0.0, 0.0])
        for key in self.slices:
            center, area = self.slices[key].calculate_center()
            tot_volume += area
            center_pos += area * center
        self.center_pos = center_pos / tot_volume
        return center_pos / tot_volume

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
        """ Based on self.slices, Dicom conours are generated for the Dicom ROI.

        :returns: Dicom ROI_CONTOURS
        """
        roi_contours = Dataset()
        contours = []
        for slice in self.slices.values():
            contours.extend(slice.create_dicom_contours())
        roi_contours.Contours = Sequence(contours)
        roi_contours.ROIDisplayColor = self.get_color(i)

        return roi_contours

    def read_vdx_old(self, content, i):
        """ Reads a single VOI from Voxelplan .vdx data from 'content', assuming a legacy .vdx format.
        VDX format 1.2.
        :params [str] content: list of lines with the .vdx content
        :params int i: line number to the list.
        :returns: current line number, after parsing the VOI.
        """
        line = content[i]
        items = line.split()
        self.name = items[1]
        self.type = int(items[3])
        i += 1
        while i < len(content):
            line = content[i]
            if re.match("voi", line) is not None:
                break
            if re.match("slice#", line) is not None:
                s = Slice(cube=self.cube)
                i = s.read_vdx_old(content, i)
                if self.cube is not None:
                    for cont1 in s.contour:
                        for cont2 in cont1.contour:
                            cont2[2] = self.cube.slice_to_z(cont2[2])  # change from slice number to mm
                if s.get_position() is None:
                    raise Exception("cannot calculate slice position")
                # TODO investigate why 100 multiplier is needed
                if self.cube is not None:
                    key = 100 * int((float(s.get_position()) - min(self.cube.slice_pos)))
                else:
                    key = 100 * int(s.get_position())
                self.slice_z.append(key)
                self.slices[key] = s
            if re.match("#TransversalObjects", line) is not None:
                pass
                # slices = int(line.split()[1]) # TODO holds information about number of skipped slices
            i += 1
        return i - 1

    def read_vdx(self, content, i):
        """ Reads a single VOI from Voxelplan .vdx data from 'content'.
        Format 2.0
        :params [str] content: list of lines with the .vdx content
        :params int i: line number to the list.
        :returns: current line number, after parsing the VOI.
        """
        line = content[i]
        self.name = ' '.join(line.split()[1:])
        number_of_slices = 10000
        i += 1
        while i < len(content):
            line = content[i]
            if re.match("key", line) is not None:
                self.key = line.split()[1]
            elif re.match("type", line) is not None:
                self.type = int(line.split()[1])
            elif re.match("number_of_slices", line) is not None:
                number_of_slices = int(line.split()[1])
            elif re.match("slice", line) is not None:
                s = Slice(cube=self.cube)
                i = s.read_vdx(content, i)
                if s.get_position() is None:
                    raise Exception("cannot calculate slice position")
                if self.cube is None:
                    raise Exception("cube not loaded")
                key = int((float(s.get_position()) - min(self.cube.slice_pos)) * 100)
                self.slice_z.append(key)
                self.slices[key] = s
            elif re.match("voi", line) is not None:
                break
            elif len(self.slices) >= number_of_slices:
                break
            i += 1
        return i - 1

    def get_roi_type_number(self, type_name):
        """
        :returns: 1 if GTV or CTV, else 0.
        """
        if type_name == 'EXTERNAL':
            return 0  # TODO: should be 10?
        elif type_name == 'AVOIDANCE':
            return 0
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
            return 'other'
        return ''

    def read_dicom(self, info, data):
        """ Reads a single ROI (= VOI) from a Dicom data set.

        :param info: (not used)
        :param Dicom data: Dicom ROI object which contains the contours.
        """
        if "Contours" not in data.dir() and "ContourSequence" not in data.dir():
            return

        self.type = self.get_roi_type_number(np.typename)
        self.color = data.ROIDisplayColor
        if "Contours" in data.dir():
            contours = data.Contours
        else:
            contours = data.ContourSequence
        for i, contour in enumerate(contours):
            key = int((float(contour.ContourData[2]) - min(self.cube.slice_pos)) * 100)
            if key not in self.slices:
                self.slices[key] = Slice(cube=self.cube)
                self.slice_z.append(key)
            self.slices[key].add_dicom_contour(contour)

    def get_thickness(self):
        """
        :returns: thickness of slice in [mm]. If there is only one slice, 3 mm is returned.
        """
        if len(self.slice_z) <= 1:
            return 3  # TODO: what is this? And shoudn't it be float?
        return abs(float(self.slice_z[1]) - float(self.slice_z[0])) / 100

    def to_voxel_string(self):
        """ Creates the Voxelplan formatted text, which can be written into a .vdx file (format 2.0).

        :returns: a str holding the all lines needed for a Voxelplan formatted file.
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
        thickness = self.get_thickness()
        for k in self.slice_z:
            sl = self.slices[k]
            pos = sl.get_position()
            out += "slice %d\n" % i
            out += "slice_in_frame %.3f\n" % pos
            out += "thickness %.3f reference " \
                   "start_pos %.3f stop_pos %.3f\n" % \
                   (thickness, pos - 0.5 * thickness, pos + 0.5 * thickness)
            out += "number_of_contours %d\n" % \
                   self.slices[k].number_of_contours()
            out += self.slices[k].to_voxel_string()
            i += 1
        return out

    def get_row_intersections(self, pos):
        """ (TODO: Documentation needed)
        """
        slice = self.get_slice_at_pos(pos[2])
        if slice is None:
            return None
        return np.sort(slice.get_intersections(pos))

    def get_slice_at_pos(self, z):
        """ Returns nearest VOI Slice at position z.

        :param float z: position z in [mm]
        :returns: a Slice object found at position z.
        """
        thickness = self.get_thickness() / 2 * 100
        for key in self.slices.keys():
            key = key
            low = z * 100 - thickness
            high = z * 100 + thickness
            if (low < key < 100 * z) or (high > key >= 100 * z):
                return self.slices[key]
        return None

    def number_of_slices(self):
        """
        :returns: number of slices covered by this VOI.
        """
        return len(self.slices)

    def concat_contour(self):
        """ Concat all contours in all slices found in this VOI.
        """
        if not self.is_concated:
            for k in self.slices.keys():
                self.slices[k].concat_contour()
        self.is_concated = True

    def get_min_max(self):
        """ Set self.temp_min and self.temp_max if they dont exist.

        :returns: minimum and maximum x y coordinates in Voi.
        """
        temp_min, temp_max = None, None
        if hasattr(self, "temp_min"):
            return self.temp_min, self.temp_max
        for key in self.slices:
            if temp_min is None:
                temp_min, temp_max = self.slices[key].get_min_max()
            else:
                min1, max1 = self.slices[key].get_min_max()
                temp_min = pytrip.res.point.min_list(temp_min, min1)
                temp_max = pytrip.res.point.max_list(temp_max, max1)
        self.temp_min = temp_min
        self.temp_max = temp_max
        return temp_min, temp_max


class Slice:
    """ The Slice class is specific for structures, and should not be confused with Slices extracted from CTX or DOS
    objects.
    """
    def __init__(self, cube=None):
        self.cube = cube
        self.contour = []

    def add_contour(self, contour):
        """ Adds a new 'contour' to the existing contours.

        :param Contour contour: the contour to be added.
        """
        self.contour.append(contour)

    def add_dicom_contour(self, dcm):
        """ Adds a Dicom CONTOUR to the existing list of contours in this Slice class.

        :param Dicom dcm: a Dicom CONTOUR object.
        """
        offset = []
        offset.append(float(self.cube.xoffset))
        offset.append(float(self.cube.yoffset))
        offset.append(float(min(self.cube.slice_pos)))
        self.contour.append(
            Contour(pytrip.res.point.array_to_point_array(np.array(dcm.ContourData, dtype=float), offset)))

    def get_position(self):
        """
        :returns: the position of this slice in [mm]
        """
        if len(self.contour) == 0:
            return None
        return self.contour[0].contour[0][2]

    def get_intersections(self, pos):
        """ (TODO: needs documentation)
        """
        intersections = []
        for c in self.contour:
            intersections.extend(pytrip.res.point.get_x_intersection(pos[1], c.contour))
        return intersections

    def calculate_center(self):
        """ Calculate the center position of all contours in this slice.

        :returns: a list of center positions [x,y,z] in [mm] for each contour found.
        """
        tot_area = 0.0
        center_pos = np.array([0.0, 0.0, 0.0])
        for contour in self.contour:
            center, area = contour.calculate_center()
            tot_area += area
            center_pos += area * center
        return center_pos / tot_area, tot_area

    def read_vdx(self, content, i):
        """ Reads a single Slice from Voxelplan .vdx data from 'content'.
        VDX format 2.0.
        :params [str] content: list of lines with the .vdx content
        :params int i: line number to the list.
        :returns: current line number, after parsing the VOI.
        """
        line = content[i]
        number_of_contours = 0
        i += 1
        while i < len(content):
            line = content[i]
            if re.match("slice_in_frame", line) is not None:
                self.slice_in_frame = float(line.split()[1])
            elif re.match("thickness", line) is not None:
                items = line.split()
                self.thickness = float(items[1])
                if len(items) == 7:
                    self.start_pos = float(items[4])
                    self.stop_pos = float(items[6])
                else:
                    self.start_pos = float(items[3])
                    self.stop_pos = float(items[5])

            elif re.match("number_of_contours", line) is not None:
                number_of_contours = int(line.split()[1])
            elif re.match("contour", line) is not None:
                c = Contour([])
                i = c.read_vdx(content, i)
                self.add_contour(c)
            elif re.match("slice", line) is not None:
                break
            elif len(self.contour) >= number_of_contours:
                break
            i += 1
        return i - 1

    def read_vdx_old(self, content, i):
        """ Reads a single Slice from Voxelplan .vdx data from 'content'.
        VDX format 1.2.
        :params [str] content: list of lines with the .vdx content
        :params int i: line number to the list.
        :returns: current line number, after parsing the VOI.
        """
        line1 = content[i]
        line2 = content[i + 1]
        line3 = content[i + 2]

        if not line1.startswith("slice#"):
            return None
        if not line2.startswith("#points"):
            return None
        if not line3.startswith("points"):
            return None

        self.slice_in_frame = float(line1.split()[1])

        c = Contour([])
        c.read_vdx_old(slice_number=self.slice_in_frame, xy_line=line3.split()[1:])
        self.add_contour(c)

        return i

    def create_dicom_contours(self):
        """ Creates and returns a list of Dicom CONTOUR objects from self.
        """

        # in order to get DICOM readable by Eclipse we need to connect each contour with CT slice
        # CT slices are identified by SOPInstanceUID
        # first we assume some default value if we cannot figure out CT slice info (i.e. CT cube is not loaded)
        ref_sop_instance_uid = '1.2.3'

        # then we check if CT cube is loaded
        if self.cube is not None:

            # if CT cube is loaded we extract DICOM representation of the cube (1 dicom per slice)
            # and select DICOM object for current slice based on slice position
            # it is time consuming as for each call of this method we generate full DICOM representation (improve!)
            candidates = [dcm for dcm in self.cube.create_dicom() if dcm.SliceLocation == self.get_position()]
            if len(candidates) > 0:
                # finally we extract CT slice SOP Instance UID
                ref_sop_instance_uid = candidates[0].SOPInstanceUID

        contour_list = []
        for item in self.contour:
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

    def to_voxel_string(self):
        """ Creates the Voxelplan formatted text, which can be written into a .vdx file (format 2.0)

        :returns: a str holding the slice information with the countour lines for a Voxelplan formatted file.
        """
        out = ""
        for i, cnt in enumerate(self.contour):
            out += "contour %d\n" % i
            out += "internal false\n"
            out += "number_of_points %d\n" % (cnt.number_of_points() + 1)
            out += cnt.to_voxel_string()
            out += "\n"
        return out

    def number_of_contours(self):
        """
        :returns: number of contours found in this Slice object.
        """
        return len(self.contour)

    def concat_contour(self):
        """ Concat all contours in this Slice object to a single contour.
        """
        for i in range(len(self.contour) - 1, 0, -1):
            self.contour[0].push(self.contour[i])
            self.contour.pop(i)
        self.contour[0].concat()

    def remove_inner_contours(self):
        """ Removes any "holes" in the contours of this slice, thereby changing the topology of the contour.
        """
        for i in range(len(self.contour) - 1, 0, -1):
            self.contour[0].push(self.contour[i])
            self.contour.pop(i)
        self.contour[0].remove_inner_contours()

    def get_min_max(self):
        """ Set self.temp_min and self.temp_max if they dont exist.

        :returns: minimum and maximum x y coordinates in Voi.
        """
        temp_min, temp_max = self.contour[0].get_min_max()
        for i in range(1, len(self.contour)):
            min1, max1 = self.contour[i].get_min_max()
            temp_min = pytrip.res.point.min_list(temp_min, min1)
            temp_max = pytrip.res.point.max_list(temp_max, max1)
        return temp_min, temp_max


class Contour:
    """ Class for handling single Contours.
    """
    def __init__(self, contour, cube=None):
        self.cube = cube
        self.children = []
        self.contour = contour

    def push(self, contour):
        """ Push a contour on the contour stack.

        :param Contour contour: a Contour object.
        """
        for i in range(len(self.children)):
            if self.children[i].contains_contour(contour):
                self.children[i].push(contour)
                return
        self.add_child(contour)

    def calculate_center(self):
        """ Calculate the center for a single contour, and the area of a contour in 3 dimensions.

        :returns: Center of the contour [x,y,z] in [mm], area [mm**2] (TODO: to be confirmed)
        """
        points = self.contour
        points.append(points[-1])
        points = np.array(points)
        dx_dy = np.array([points[i + 1] - points[i] for i in range(len(points) - 1)])
        if abs(points[0, 2] - points[1, 2]) < 0.01:
            area = -sum(points[0:len(points) - 1, 1] * dx_dy[:, 0])
            paths = np.array((dx_dy[:, 0]**2 + dx_dy[:, 1]**2)**0.5)
        elif abs(points[0, 1] - points[1, 1]) < 0.01:
            area = -sum(points[0:len(points) - 1, 2] * dx_dy[:, 0])
            paths = np.array((dx_dy[:, 0]**2 + dx_dy[:, 2]**2)**0.5)
        elif abs(points[0, 0] - points[1, 0]) < 0.01:
            area = -sum(points[0:len(points) - 1, 2] * dx_dy[:, 1])
            paths = np.array((dx_dy[:, 1]**2 + dx_dy[:, 2]**2)**0.5)
        total_path = sum(paths)

        center = np.array([sum(points[0:len(points) - 1, 0] * paths) / total_path,
                           sum(points[0:len(points) - 1:, 1] * paths) / total_path,
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

    def to_voxel_string(self):
        """ Creates the Voxelplan formatted text, which can be written into a .vdx file.

        :returns: a str holding the contour points needed for a Voxelplan formatted file.
        """
        out = ""
        for i, cnt in enumerate(self.contour):
            out += " %.3f %.3f %.3f %.3f %.3f %.3f\n" % (cnt[0], cnt[1], cnt[2], 0, 0, 0)
        out += " %.3f %.3f %.3f %.3f %.3f %.3f\n" % (self.contour[0][0], self.contour[0][1], self.contour[0][2],
                                                     0, 0, 0)
        return out

    def read_vdx(self, content, i):
        """ Reads a single Contour from Voxelplan .vdx data from 'content'.
        VDX format 2.0.

        :params [str] content: list of lines with the .vdx content
        :params int i: line number to the list.
        :returns: current line number, after parsing the VOI.
        """
        set_point = False
        points = 0
        j = 0
        while i < len(content):
            line = content[i]
            if set_point:
                if j >= points - 1:
                    break
                con_dat = line.split()
                self.contour.append([float(con_dat[0]), float(con_dat[1]), float(con_dat[2])])
                j += 1
            else:
                if re.match("internal_false", line) is not None:
                    self.internal_false = True
                if re.match("number_of_points", line) is not None:
                    points = int(line.split()[1])
                    set_point = True
            i += 1
        return i - 1

    def read_vdx_old(self, slice_number, xy_line):
        """ Reads a single Contour from Voxelplan .vdx data from 'content' and appends it to self.contour data
        VDX format 1.2.
        :params slice_number: list of numbers (as characters) with slice number
        :params xy_line: list of numbers (as characters) representing X and Y coordinates of a contour
        """

        # and example of xy_line: 3021 4761 2994 4899 2916 5015
        xy_pairs = [xy_line[i:i + 2] for i in range(0, len(xy_line), 2)]  # make list of pairs
        for x, y in xy_pairs:
            # TRiP98 saves X,Y coordinates as integers, to get [mm] they needs to be divided by 16
            self.contour.append([float(x) / 16.0, float(y) / 16.0, float(slice_number)])

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
        """ Print child to stdout.

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
        """ In case of multiple contours in the same slice, this method will concat them to a single conour.
        This is important for TRiP98 compability, as TRiP98 cannot handle multiple contours in the same slice of
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
        """ Merge two contours into a single one.
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
