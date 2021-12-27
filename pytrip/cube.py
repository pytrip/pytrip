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
This module provides the Cube class, which is used by the CTX, DOS, LET and VDX modules.
A cube is a 3D object holding data, such as CT Hounsfield units, Dose- or LET values.
"""
import os
import io
import re
import sys
import logging
import datetime

import numpy as np

try:
    # as of version 1.0 pydicom package import has beed renamed from dicom to pydicom
    from pydicom import uid
    from pydicom.dataset import Dataset, FileDataset
    _dicom_loaded = True
except ImportError:
    try:
        # fallback to old (<1.0) pydicom package version
        from dicom import UID as uid  # old pydicom had UID instead of uid
        from dicom.dataset import Dataset, FileDataset
        _dicom_loaded = True
    except ImportError:
        _dicom_loaded = False

from pytrip.error import InputError, ModuleNotLoadedError, FileNotFound
from pytrip.util import TRiP98FilePath, TRiP98FileLocator

logger = logging.getLogger(__name__)


class Cube(object):
    """ Top level class for 3-dimensional data cubes used by e.g. DosCube, CtxCube and LETCube.
    Otherwise, this cube class may be used for storing different kinds of data, such as number of cells,
    oxygenation level, surviving fraction, etc.
    """

    header_file_extension = '.hed'
    data_file_extension = None
    allowed_suffix = ()

    def __init__(self, cube=None):
        self.pydata_type = np.int16
        if cube is not None:  # copying constructor
            self.header_set = cube.header_set
            self.version = cube.version
            self.modality = cube.modality
            self.created_by = cube.created_by
            self.creation_info = cube.creation_info
            self.primary_view = cube.primary_view
            self.data_type = cube.data_type
            self.num_bytes = cube.num_bytes
            self.byte_order = cube.byte_order
            self.patient_name = cube.patient_name
            self.patient_id = cube.patient_id
            self.slice_dimension = cube.slice_dimension
            self.pixel_size = cube.pixel_size
            self.slice_distance = cube.slice_distance
            self.slice_thickness = cube.slice_thickness
            self.slice_number = cube.slice_number
            self.xoffset = cube.xoffset  # self.xoffset are in mm, synced with DICOM contours
            self.dimx = cube.dimx
            self.yoffset = cube.yoffset  # self.yoffset are in mm, synced with DICOM contours
            self.dimy = cube.dimy
            self.zoffset = cube.zoffset  # self.zoffset are in mm, synced with DICOM contours
            self.dimz = cube.dimz
            self.z_table = cube.z_table
            self.slice_pos = cube.slice_pos
            self.basename = cube.basename
            self._set_format_str()
            self._set_number_of_bytes()

            # unique for whole structure set
            self._dicom_study_instance_uid = cube._dicom_study_instance_uid
            self._ct_dicom_series_instance_uid = cube._ct_dicom_series_instance_uid

            # unique for each CT slice
            self._ct_sop_instance_uid = cube._ct_sop_instance_uid

            self.cube = np.zeros((self.dimz, self.dimy, self.dimx), dtype=cube.pydata_type)  # skipcq PTC-W0052

        else:
            import getpass
            from pytrip import __version__ as _ptversion

            self.header_set = False
            self.version = "2.0"
            self.modality = "CT"
            try:
                self.created_by = getpass.getuser()
            except ImportError:
                # it may happen that on Windows system `getpass.getuser` won't work
                # this will manifest as "ModuleNotFoundError/ImportError: No module named 'pwd'" exception
                # as the getpass is trying to get the login from the password database which relies
                # on systems which support the pwd module
                # in such case we set created_by field to a fixed value
                self.created_by = 'pytrip'
            self.creation_info = "Created with PyTRiP98 {:s}".format(_ptversion)
            self.primary_view = "transversal"  # e.g. transversal
            self.data_type = ""
            self.num_bytes = ""
            self.byte_order = "vms"  # aix or vms
            self.patient_name = ""
            self.patient_id = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')  # create a new patient ID if absent
            self.slice_dimension = ""  # eg. 256 meaning 256x256 pixels.
            self.pixel_size = ""  # size in [mm]
            self.slice_distance = ""  # distance between slices in [mm]
            self.slice_thickness = ""  # thickness of slice (usually equal to slice_distance) in [mm]
            self.slice_number = ""  # number of slices in file.
            self.xoffset = 0.0
            self.dimx = ""  # number of pixels along x (e.g. 256)
            self.yoffset = 0.0
            self.dimy = ""
            self.zoffset = 0.0
            self.dimz = ""
            self.slice_pos = []
            self.basename = ""

            # UIDs unique for whole structure set
            # generation of UID is done here in init, the reason why we are not generating them in create_dicom
            # method is that subsequent calls to write method shouldn't changed UIDs
            self._dicom_study_instance_uid = uid.generate_uid(prefix=None)
            self._ct_dicom_series_instance_uid = uid.generate_uid(prefix=None)

            # unique for each CT slice
            self._ct_sop_instance_uid = uid.generate_uid(prefix=None)

            self.z_table = False  # positions are stored in self.slice_pos (list of slice#,pos(mm),thickness(mm),tilt)

    def __str__(self):
        return "Cube: "+self.basename

    def __add__(self, other):
        """ Overload + operator
        """
        c = type(self)(self)
        if Cube in other.__class__.__bases__:
            c.cube = other.cube + self.cube
        else:
            c.cube = self.cube + float(other)
        return c

    def __sub__(self, other):
        """ Overload - operator
        """
        c = type(self)(self)
        if Cube in other.__class__.__bases__:
            c.cube = self.cube - other.cube
        else:
            c.cube = self.cube - float(other)
        return c

    def __mul__(self, other):
        """ Overload * operator
        """
        c = type(self)(self)
        if Cube in other.__class__.__bases__:
            c.cube = other.cube * self.cube
        else:
            t = type(c.cube[0, 0, 0])
            c.cube = np.array(self.cube * float(other), dtype=t)
        return c

    def __div__(self, other):
        """ Overload / operator
        """
        c = type(self)(self)
        if Cube in other.__class__.__bases__:
            c.cube = self.cube / other.cube
        else:
            t = type(c.cube[0, 0, 0])
            c.cube = np.array(self.cube / float(other), dtype=t)
        c.cube[np.isnan(c.cube)] = 0  # fix division by zero NaNs
        return c

    __truediv__ = __div__

    # TODO __floordiv__ should also be handled

    def is_compatible(self, other):
        """ Check if this Cube object is compatible in size and dimensions with 'other' cube.

        A cube object can be a CtxCube, DosCube, LETCube or similar object.
        Unlike check_compatibility(), this function compares itself to the other cube.

        :param Cube other: The other Cube object which will be checked compatibility with.
        :returns: True if compatible.
        """
        return self.check_compatibility(self, other)

    @staticmethod
    def check_compatibility(a, b):
        """
        Simple comparison of cubes. if X,Y,Z dims are the same, and
        voxel sizes as well, then they are compatible. (Duck typed)

        See also the function is_compatible().

        :params Cube a: the first cube to be compared with the second (b).
        :params Cube b: the second cube to be compared with the first (a).

        """
        eps = 1e-5

        x_dim_compatible = (a.dimx == b.dimx)
        y_dim_compatible = (a.dimy == b.dimy)
        z_dim_compatible = (a.dimz == b.dimz)
        pixel_size_compatible = (a.pixel_size - b.pixel_size <= eps)
        slice_distance_compatible = (a.slice_distance == b.slice_distance)
        return x_dim_compatible and y_dim_compatible and z_dim_compatible and pixel_size_compatible and \
            slice_distance_compatible

    def indices_to_pos(self, indices):
        """ Translate index number of a voxel to real position in [mm], including any offsets.

        The z position is always following the slice positions.

        :params [int] indices: tuple or list of integer indices (i,j,k) or [i,j,k]
        :returns: list of positions, including offsets, as a list of floats [x,y,z]
        """
        pos = [(indices[0] + 0.5) * self.pixel_size + self.xoffset, (indices[1] + 0.5) * self.pixel_size + self.yoffset,
               self.slice_pos[indices[2]]]
        logger.debug("Map [i,j,k] {:d} {:d} {:d} to [x,y,z] {:.2f} {:.2f} {:.2f}".format(
            indices[0], indices[1], indices[2], pos[0], pos[1], pos[2]))
        return pos

    def slice_to_z(self, slice_number):
        """ Return z-position in [mm] of slice number (starting at 1).

        :params int slice_number: slice number, starting at 1 and no bound check done here.
        :returns: position of slice in [mm] including offset
        """
        # note that self.slice_pos contains an array of positions including any zoffset.
        return self.slice_pos[slice_number - 1]

    def mask_by_voi_all(self, voi, preset=0, data_type=np.int16):
        """ Attaches/overwrites Cube.data based on a given Voi.

        Voxels within the structure are filled it with 'preset' value.
        Voxels outside the contour will be filled with Zeros.

        :param Voi voi: the volume of interest
        :param int preset: value to be assigned to the voxels within the contour.
        :param data_type: numpy data type, default is np.int16

        TODO: this needs some faster implementation.
        """
        data = np.array(np.zeros((self.dimz, self.dimy, self.dimx)), dtype=data_type)
        if preset != 0:
            for i_z in range(self.dimz):
                for i_y in range(self.dimy):
                    # For a line along y, figure out how many contour intersections there are,
                    # then check how many intersections there are with x < than current point.
                    # If the number is odd, then the point is inside the VOI.
                    # If the number is even, then the point is outisde the VOI.
                    # This algorithm also works with multiple disconnected contours.
                    intersection = voi.get_row_intersections(self.indices_to_pos([0, i_y, i_z]))
                    if intersection is None:
                        break
                    if len(intersection) > 0:
                        k = 0
                        for i_x in range(self.dimx):
                            # count the number of intersections k along y, where intersection_x < current x position
                            if self.indices_to_pos([i_x, 0, 0])[0] > intersection[k]:
                                k += 1
                                if k >= len(intersection):
                                    break
                            if k % 2 == 1:  # voxel is inside structure, if odd number of intersections.
                                data[i_z][i_y][i_x] = preset
        self.cube = data

    def create_empty_cube(self, value, dimx, dimy, dimz, pixel_size, slice_distance, xoffset=0.0, yoffset=0.0,
                          slice_offset=0.0):
        """ Creates an empty Cube object.

        Values are stored as 2-byte integers.

        :param int16 value: integer value which will be assigned to all voxels.
        :param int dimx: number of voxels along x
        :param int dimy: number of voxels along y
        :param int dimz: number of voxels along z
        :param float pixel_size: size of each pixel (x == y) in [mm]
        :param float slice_distance: the distance between two slices (z) in [mm]
        :param float xoffset: offset in X in [mm] (default 0.0 mm)
        :param float yoffset: offset in Y in [mm] (default 0.0 mm)
        :param float slice_offset: start position of the first slice (offset in Z) in [mm] (default 0.0 mm)
        """
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.slice_number = dimz
        self.pixel_size = pixel_size
        self.slice_distance = slice_distance
        self.slice_thickness = slice_distance  # use distance for thickness as default
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.zoffset = slice_offset
        self.cube = np.ones((dimz, dimy, dimx), dtype=np.int16) * value
        self.slice_dimension = dimx
        self.num_bytes = 2
        self.data_type = "integer"
        self.slice_pos = [slice_distance * i + slice_offset for i in range(dimz)]
        self.header_set = True
        self.patient_id = ''
        # UIDs unique for whole structure set
        # generation of UID is done here in init, the reason why we are not generating them in create_dicom
        # method is that subsequent calls to write method shouldn't changed UIDs
        self._dicom_study_instance_uid = uid.generate_uid(prefix=None)
        self._ct_dicom_series_instance_uid = uid.generate_uid(prefix=None)
        # unique for each CT slice
        self._ct_sop_instance_uid = uid.generate_uid(prefix=None)

    def mask_by_voi(self, voi, value):
        """ Overwrites the Cube voxels within the given Voi with 'value'.

        Voxels within the structure are filled it with 'value'.
        Voxels outside the contour are not touched.

        :param Voi voi: the volume of interest
        :param value=0: value to be assigned to the voxels within the contour.
        """
        for i_z in range(self.dimz):
            for i_y in range(self.dimy):
                intersection = voi.get_row_intersections(self.indices_to_pos([0, i_y, i_z]))
                if intersection is None:
                    break
                if len(intersection) > 0:
                    k = 0
                    for i_x in range(self.dimx):
                        if self.indices_to_pos([i_x, 0, 0])[0] > intersection[k]:
                            k += 1
                            if k >= (len(intersection)):
                                break
                        if k % 2 == 1:  # voxel is inside structure, if odd number of intersections.
                            self.cube[i_z][i_y][i_x] = value

    def mask_by_voi_add(self, voi, value=0):
        """ Add 'value' to all voxels within the given Voi

        'value' is added to each voxel value within the given volume of interest.
        Voxels outside the volume of interest are not touched.

        :param Voi voi: the volume of interest
        :param value=0: value to be added to the voxel values within the contour.
        """
        for i_z in range(self.dimz):
            for i_y in range(self.dimy):
                intersection = voi.get_row_intersections(self.indices_to_pos([0, i_y, i_z]))
                if intersection is None:
                    break
                if len(intersection) > 0:
                    k = 0
                    for i_x in range(self.dimx):
                        if self.indices_to_pos([i_x, 0, 0])[0] > intersection[k]:
                            k += 1
                            if k >= (len(intersection)):
                                break
                        if k % 2 == 1:  # voxel is inside structure, if odd number of intersections.
                            self.cube[i_z][i_y][i_x] += value

    def merge(self, cube):
        self.cube = np.maximum(self.cube, cube.cube)

    def merge_zero(self, cube):
        self.cube[self.cube == 0] = cube.cube[self.cube == 0]

    # ######################  READING TRIP98 FILES #######################################

    @classmethod
    def header_file_name(cls, path_name):
        return TRiP98FilePath(path_name, cls).header

    @classmethod
    def data_file_name(cls, path_name):
        return TRiP98FilePath(path_name, cls).datafile

    def read(self, path):
        """
        Reads both TRiP98 data and its associated header into the Cube object.

        Cube can be read providing a filename stripped of extension, i.e:
        >>> import pytrip as pt
        >>> c1 = pt.CtxCube()
        >>> c1.read("tests/res/TST003/tst003000")

        We can also read header file and data path which do not share common basename:
        >>> c2 = pt.CtxCube()
        >>> c2.read(("tests/res/TST003/tst003012.hed", "tests/res/TST003/tst003000.ctx.gz"))

        :param path: string or sequence of strings (length 2)
        :return:
        """

        # let us check if path is a string in a way python 2 and 3 will like it
        # based on https://stackoverflow.com/questions/4843173/how-to-check-if-type-of-a-variable-is-string
        running_python2 = sys.version_info.major == 2
        path_string = isinstance(path, (str, bytes) if not running_python2 else basestring)  # NOQA: F821

        # single argument of string type, i.e. filename without extension
        if path_string:
            self.basename = os.path.basename(TRiP98FilePath(path, self).basename)

            path_locator = TRiP98FileLocator(path, self)

            header_path = path_locator.header
            datafile_path = path_locator.datafile

            if not datafile_path or not header_path:
                raise FileNotFound("Loading {:s} failed, file not found".format(path))

        # tuple with path to header and datafile
        elif len(path) == 2:
            header_path, datafile_path = path

            # security checks for header file
            # first check - validity of the path
            if not TRiP98FilePath(header_path, self).is_valid_header_path():
                logger.warning("Loading {:s} which doesn't look like valid header path".format(header_path))

            # second check - if file exists
            if not os.path.exists(header_path):
                header_path_locator = TRiP98FileLocator(header_path, self)
                if header_path_locator.header is not None:
                    logger.warning("Did you meant to load {:s}, instead of {:s} ?".format(
                        header_path_locator.header, header_path))
                raise FileNotFound("Loading {:s} failed, file not found".format(header_path))

            # security checks for datafile path
            # first check - validity of the path
            if not TRiP98FilePath(datafile_path, self).is_valid_datafile_path():
                logger.warning("Loading {:s} which doesn't look like valid datafile path".format(datafile_path))

            # second check - if file exists
            if not os.path.exists(datafile_path):
                datafile_path_locator = TRiP98FileLocator(datafile_path, self)
                if datafile_path_locator.datafile is not None:
                    logger.warning("Did you meant to load {:s}, instead of {:s} ?".format(
                        datafile_path_locator.datafile, datafile_path))
                raise FileNotFound("Loading {:s} failed, file not found".format(datafile_path))

            self.basename = ""  # TODO user may provide two completely different filenames for header and datafile
            # i.e. read( ("1.hed", "2.dos"), what about basename then ?

        else:
            raise ValueError("More than two arguments provided as path variable to Cube.read method")

        # finally read files
        self._read_trip_header_file(header_path=header_path)
        self._read_trip_data_file(datafile_path=datafile_path, header_path=header_path)

    def _read_trip_header_file(self, header_path):  # TODO: could be made private? #126
        """ Reads a header file, accepts also if file is .gz compressed.
        First the un-zipped files will be attempted to read.
        Should these not exist, then the .gz are attempted.

        However, if the .hed.gz file was explicitly stated,
        then this file will also be loaded, even if a .hed is available.
        """

        # sanity check
        if header_path is not None:
            logger.info("Reading header file" + header_path)
        else:
            raise IOError("Could not find file " + header_path)

        # on python 3 read text, but on python 2 read bytes
        file_open_mode = "rt"
        if sys.version_info.major == 2:
            file_open_mode = "rb"

        # load plain of gzipped file
        content = ""
        if header_path.endswith(".gz"):
            import gzip
            with gzip.open(header_path, file_open_mode) as fp:
                content = fp.read()
        else:
            with open(header_path, file_open_mode) as fp:
                content = fp.read()
        if sys.version_info.major == 2:
            # if it encounters unknown char then replace it with replacement character
            content = content.decode("utf-8", errors="replace")

        # fill self with data
        self._parse_trip_header(content)
        self._set_format_str()
        logger.debug("Format string:" + self.format_str)

    def _read_trip_data_file(self,
                             datafile_path,
                             header_path,
                             multiply_by_2=False):  # TODO: could be made private? #126
        """Read TRiP98 formatted data.

        If header file was not previously loaded, it will be attempted first.

        Due to an issue in VIRTUOS, sometimes DosCube data have been reduced with a factor of 2.
        Setting multiply_by_2 to True, will restore the true values, in this case.

        :param datafile_path: Path to TRiP formatted data.
        :param multiply_by_2: The data read will automatically be multiplied with a factor of 2.
        """

        # fill header data if self.header is empty
        if not self.header_set:
            self._read_trip_header_file(header_path)

        # raise exception if reading header failed
        if not self.header_set:
            raise InputError("Header file not loaded")

        # preparation
        data_dtype = np.dtype(self.format_str)
        data_count = self.dimx * self.dimy * self.dimz

        # load data from data file (gzipped or not)
        logger.info("Opening file: " + datafile_path)
        if datafile_path.endswith('.gz'):
            import gzip
            with gzip.open(datafile_path, "rb") as f:
                s = f.read(data_dtype.itemsize * data_count)
                tmpcube = np.frombuffer(s, dtype=data_dtype, count=data_count)
                # frombuffer returns read-only array, so we need to make it writable
                cube = np.require(tmpcube, dtype=data_dtype, requirements=['W', 'O'])
        else:
            cube = np.fromfile(datafile_path, dtype=data_dtype)

        if self.byte_order == "aix":
            logger.info("AIX big-endian data.")
            # byteswapping is not needed anymore, handled by "<" ">" in dtype

        # sanity check
        logger.info("Cube data points : {:d}".format(len(cube)))
        if len(cube) != self.dimx * self.dimy * self.dimz:
            logger.error("Header size and cube size dont match.")
            logger.error("Cube data points : {:d}".format(len(cube)))
            logger.error("Header says      : {:d} = {:d} * {:d} * {:d}".format(self.dimx * self.dimy * self.dimz,
                                                                               self.dimx, self.dimy, self.dimz))
            raise IOError("Header data and dose cube size are not consistent.")

        cube = np.reshape(cube, (self.dimz, self.dimy, self.dimx))
        if multiply_by_2:
            logger.warning("Cube was previously rescaled to 50%. Now multiplying with 2.")
            cube *= 2
        self.cube = cube

    def _parse_trip_header(self, content):
        """ Parses content which was read from a trip header.
        """
        i = 0
        self.header_set = True
        content = content.split('\n')
        self.z_table = False
        while i < len(content):
            if re.match("version", content[i]):
                self.version = content[i].split()[1]
            if re.match("modality", content[i]):
                self.modality = content[i].split()[1]
            if re.match("created_by", content[i]):
                self.created_by = content[i].replace("created_by ", "", 1)
                self.created_by = self.created_by.rstrip()
            if re.match("creation_info", content[i]):
                self.creation_info = content[i].replace("creation_info ", "", 1)
                self.creation_info = self.creation_info.rstrip()
            if re.match("primary_view", content[i]):
                self.primary_view = content[i].split()[1]
            if re.match("data_type", content[i]):
                self.data_type = content[i].split()[1]
            if re.match("num_bytes", content[i]):
                self.num_bytes = int(content[i].split()[1])
            if re.match("byte_order", content[i]):
                self.byte_order = content[i].split()[1]
            if re.match("patient_name", content[i]):
                self.patient_name = content[i].split()[1]
            if re.match("slice_dimension", content[i]):
                self.slice_dimension = int(content[i].split()[1])
            if re.match("pixel_size", content[i]):
                self.pixel_size = float(content[i].split()[1])
            if re.match("slice_distance", content[i]):
                self.slice_distance = float(content[i].split()[1])
                self.slice_thickness = float(content[i].split()[1])  # TRiP format only. See #342
            if re.match("slice_number", content[i]):
                self.slice_number = int(content[i].split()[1])
            if re.match("xoffset", content[i]):
                self.xoffset = int(content[i].split()[1])
            if re.match("yoffset", content[i]):
                self.yoffset = int(content[i].split()[1])
            if re.match("zoffset", content[i]):
                self.zoffset = int(content[i].split()[1])
            if re.match("dimx", content[i]):
                self.dimx = int(content[i].split()[1])
            if re.match("dimy", content[i]):
                self.dimy = int(content[i].split()[1])
            if re.match("dimz", content[i]):
                self.dimz = int(content[i].split()[1])
            if re.match("slice_no", content[i]):
                self.slice_pos = [float(j) for j in range(self.slice_number)]
                self.z_table = True
                i += 1
                for j in range(self.slice_number):
                    self.slice_pos[j] = float(content[i].split()[1])
                    i += 1
            i += 1

        # zoffset from TRiP contains the integer amount of slice thicknesses as offset.
        # Here we convert to an actual offset in mm, which is stored in self
        self.xoffset *= self.pixel_size
        self.yoffset *= self.pixel_size
        self.zoffset *= self.slice_distance

        # if zoffset is 0 and it shouldn't be, then calculate it
        if self.slice_pos and self.slice_pos[0] and self.zoffset == 0:
            self.zoffset = self.slice_pos[0]

        logger.debug("TRiP loaded offsets: {:f} {:f} {:f}".format(self.xoffset, self.yoffset, self.zoffset))

        # generate slice position tables, if absent in header file
        # Note:
        # - ztable in .hed is _without_ offset
        # - self.slice_pos however holds values _including_ offset.
        if not self.z_table:
            self.slice_pos = [self.zoffset + _i * self.slice_distance for _i in range(self.slice_number)]
        self._set_format_str()

    def _set_format_str(self):
        """Set format string according to byte_order.
        """
        if self.byte_order == "vms":
            self.format_str = "<"
        elif self.byte_order == "aix":
            self.format_str = ">"
        self._set_number_of_bytes()

    def _set_number_of_bytes(self):
        """Set format_str and pydata_type according to num_bytes and data_type
        """
        if self.data_type == "integer":
            if self.num_bytes == 1:
                self.format_str += "b"
                self.pydata_type = np.int8
            if self.num_bytes == 2:
                self.format_str += "h"
                self.pydata_type = np.int16
            if self.num_bytes == 4:
                self.format_str += "i"
                self.pydata_type = np.int32
        elif self.data_type in ["float", "double"]:
            if self.num_bytes == 4:
                self.format_str += "f"
                self.pydata_type = np.float32
            if self.num_bytes == 8:
                self.format_str += "d"
                self.pydata_type = np.double
        else:
            print("Format:", self.byte_order, self.data_type, self.num_bytes)
            raise IOError("Unsupported format.")
        logger.debug("self.format_str: '{}'".format(self.format_str))

    # ######################  WRITING TRIP98 FILES #######################################

    def write(self, path):
        """Write the Cube and its header to a file with the filename 'path'.

        :param str path: path to header file, data file or basename (without extension)
        :returns: tuple header_path, datafile_path: paths to header file and datafiles where data was saved
        (may be different from input path if user provided a partial basename)
        """

        running_python2 = sys.version_info.major == 2
        path_string = isinstance(path, (str, bytes) if not running_python2 else basestring)  # NOQA: F821

        if path_string:
            header_path = self.header_file_name(path)
            datafile_path = self.data_file_name(path)

        elif len(path) == 2:
            header_path, datafile_path = path

            # security checks for header file
            # first check - validity of the path
            if not TRiP98FilePath(header_path, self).is_valid_header_path():
                logger.warning("Loading {:s} which doesn't look like valid header path".format(header_path))

            # security checks for datafile path
            # first check - validity of the path
            if not TRiP98FilePath(datafile_path, self).is_valid_datafile_path():
                logger.warning("Loading {:s} which doesn't look like valid datafile path".format(datafile_path))
        else:
            raise ValueError("More than two arguments provided as path variable to Cube.write method")

        # finally write files
        self._write_trip_header(header_path)
        self._write_trip_data(datafile_path)

        return header_path, datafile_path

    def _write_trip_header(self, path):
        """ Write a TRiP98 formatted header file, based on the available meta data.

        :param path: fully qualified path, including file extension (.hed)
        """
        from distutils.version import LooseVersion
        output_str = "version " + self.version + "\n"
        output_str += "modality " + self.modality + "\n"
        # include created_by and creation_info only for files newer than 1.4
        if LooseVersion(self.version) >= LooseVersion("1.4"):
            output_str += "created_by {:s}\n".format(self.created_by)
            output_str += "creation_info {:s}\n".format(self.creation_info)
        output_str += "primary_view " + self.primary_view + "\n"
        output_str += "data_type " + self.data_type + "\n"
        output_str += "num_bytes " + str(self.num_bytes) + "\n"
        output_str += "byte_order " + self.byte_order + "\n"
        if self.patient_name == "":
            self.patient_name = "Anonymous"
        # patient_name in .hed must be equal to the base filename without extension, else TRiP98 wont import VDX
        _fname = os.path.basename(path)
        _pname = os.path.splitext(_fname)[0]
        output_str += "patient_name {:s}\n".format(_pname)
        output_str += "slice_dimension {:d}\n".format(self.slice_dimension)
        output_str += "pixel_size {:.7f}\n".format(self.pixel_size)
        output_str += "slice_distance {:.7f}\n".format(self.slice_distance)
        output_str += "slice_number " + str(self.slice_number) + "\n"
        output_str += "xoffset {:d}\n".format(int(round(self.xoffset / self.pixel_size)))
        output_str += "dimx {:d}\n".format(self.dimx)
        output_str += "yoffset {:d}\n".format(int(round(self.yoffset / self.pixel_size)))
        output_str += "dimy {:d}\n".format(self.dimy)

        # zoffset in Voxelplan .hed seems to be broken, and should not be used if not = 0
        # to apply zoffset, z_table should be used instead.
        # This means, self.zoffset should not be used anywhere.
        output_str += "zoffset 0\n"
        output_str += "dimz " + str(self.dimz) + "\n"
        if self.z_table:
            output_str += "z_table yes\n"
            output_str += "slice_no  position  thickness  gantry_tilt\n"
            for i, item in enumerate(self.slice_pos):
                output_str += "  {:<3d}{:14.4f}{:13.4f}{:14.4f}\n".format(i + 1, item, self.slice_thickness,
                                                                          0)  # 0 gantry tilt
        else:
            output_str += "z_table no\n"

        # for compatibility with python 2.7 we need to use `io.open` instead of `open`,
        # as `open` function in python 2.7 cannot handle `newline` argument.
        # This needs to be followed by `decode()`d string being written
        with io.open(path, "w+", newline='\n') as f:
            try:
                f.write(output_str)
            except TypeError:
                f.write(output_str.decode())

    def _write_trip_data(self, path):
        """ Writes the binary data cube in TRiP98 format to a file.

        Type is specified by self.pydata_type and self.byte_order attributes.

        :param str path: Full path including file extension.
        """
        cube = np.array(self.cube, dtype=self.pydata_type)
        if self.byte_order == "aix":
            cube = cube.byteswap()
        cube.tofile(path)

    # ######################  READING DICOM FILES #######################################

    def _set_z_table_from_dicom(self, dcm):
        """ Creates the slice position lookup table based on a given Dicom object.
        The table is attached to self.

        :param DICOM dcm: DICOM object provided by pydicom.
        """
        # TODO: can we rely on that this will always be sorted?
        # if yes, then all references to whether this is sorted or not can be removed hereafter
        # (see also pytripgui) /NBassler
        self.slice_pos = []
        for dcm_image in dcm["images"]:
            self.slice_pos.append(float(dcm_image.ImagePositionPatient[2]))

    def _set_header_from_dicom(self, dcm):
        """ Creates the header metadata for this Cube class, based on a given Dicom object.

        :param DICOM dcm: Dicom object which will be used for generating the header data.
        """
        if not _dicom_loaded:
            raise ModuleNotLoadedError("Dicom")
        ds = dcm["images"][0]
        self.version = "1.4"
        self.created_by = "pytrip"
        self.creation_info = "Created by PyTRiP98;"
        self.primary_view = "transversal"
        self.set_data_type(type(ds.pixel_array[0][0]))
        self.patient_name = ds.PatientName
        self.patient_id = ds.PatientID
        self.basename = ds.PatientID.replace(" ", "_")
        self.slice_dimension = int(ds.Rows)  # should be changed ?
        self.pixel_size = float(ds.PixelSpacing[0])  # (0028, 0030) Pixel Spacing (DS)
        self.slice_thickness = ds.SliceThickness  # (0018, 0050) Slice Thickness (DS)
        # slice_distance != SliceThickness. One may have overlapping slices. See #342
        self.slice_number = len(dcm["images"])
        self.xoffset = float(ds.ImagePositionPatient[0])
        self.dimx = int(ds.Rows)  # (0028, 0010) Rows (US)
        self.yoffset = float(ds.ImagePositionPatient[1])
        self.dimy = int(ds.Columns)  # (0028, 0011) Columns (US)
        self.zoffset = float(ds.ImagePositionPatient[2])  # note that zoffset should not be used.
        self.dimz = len(dcm["images"])
        self._set_z_table_from_dicom(dcm)
        self.z_table = True

        # Fix for bug #342
        # TODO: slice_distance should probably be a list of distances,
        # but for now we will just use the distance between the first two slices.
        if len(self.slice_pos) > 1:  # _set_z_table_from_dicom() must be called before
            self.slice_distance = abs(self.slice_pos[1] - self.slice_pos[0])
            logger.debug("Slice distance set to {:.2f}".format(self.slice_distance))
        else:
            logger.warning("Only a single slice found. Setting slice_distance to slice_thickness.")
            self.slice_distance = self.slice_thickness

        if self.slice_thickness > self.slice_distance:
            # TODO: this is probably valid dicom format, however let's print a warning for now
            # as it may indicate some problem with the input dicom, as it is rather unusual.
            logger.warning("Overlapping slices found: slice thickness is larger than the slice distance.")

        self.set_byteorder()
        self.data_type = "integer"
        self.num_bytes = 2
        self._set_format_str()
        self.header_set = True

        # unique for whole structure set
        self._dicom_study_instance_uid = ds.StudyInstanceUID
        self._ct_dicom_series_instance_uid = ds.SeriesInstanceUID

    def set_byteorder(self, endian=None):
        """Set/change the byte order of the data to be written to disk.

        Available options are:
        - 'little' vms, Intel style little-endian byte order.
        - 'big' aix, Motorola style big-endian byte order.
        - if unspecified, the native system dependent endianess is used.

        :param str endian: optional string containing the endianess.
        """
        if endian is None:
            endian = sys.byteorder
        if endian == 'little':
            self.byte_order = "vms"
        elif endian == 'big':
            self.byte_order = "aix"
        else:
            raise ValueError("set_byteorder error: unknown endian " + str(endian))

    def set_data_type(self, data_type):
        """ Sets the data type for the TRiP98 header files.

        :param numpy.type data_type: numpy type, e.g. np.uint16
        """
        if data_type is np.int8 or data_type is np.uint8:
            self.data_type = "integer"
            self.num_bytes = 1
        elif data_type is np.int16 or data_type is np.uint16:
            self.data_type = "integer"
            self.num_bytes = 2
        elif data_type is np.int32 or data_type is np.uint32:
            self.data_type = "integer"
            self.num_bytes = 4
        elif data_type is np.float:
            self.data_type = "float"
            self.num_bytes = 4
        elif data_type is np.double:
            self.data_type = "double"
            self.num_bytes = 8

    # ######################  WRITING DICOM FILES #######################################

    def create_dicom_base(self):
        if _dicom_loaded is False:
            raise ModuleNotLoadedError("Dicom")
        if self.header_set is False:
            raise InputError("Header not loaded")

        # TODO tags + code datatypes are described here:
        # https://www.dabsoft.ch/dicom/6/6/#(0020,0012)
        # datatype codes are described here:
        # ftp://dicom.nema.org/medical/DICOM/2013/output/chtml/part05/sect_6.2.html

        meta = Dataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        # Media Storage SOP Instance UID tag 0x0002,0x0003 (type UI - Unique Identifier)
        meta.MediaStorageSOPInstanceUID = self._ct_sop_instance_uid
        meta.ImplementationClassUID = "1.2.3.4"
        meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian  # Implicit VR Little Endian - Default Transfer Syntax
        ds = FileDataset("file", {}, file_meta=meta, preamble=b"\0" * 128)
        ds.PatientName = self.patient_name
        if self.patient_id in (None, ''):
            ds.PatientID = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
        else:
            ds.PatientID = self.patient_id  # Patient ID tag 0x0010,0x0020 (type LO - Long String)
        ds.PatientSex = ''  # Patient's Sex tag 0x0010,0x0040 (type CS - Code String)
        #                      Enumerated Values: M = male F = female O = other.
        ds.PatientBirthDate = '19010101'
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.AccessionNumber = ''
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.SOPClassUID = '1.2.3'  # !!!!!!!!
        # SOP Instance UID tag 0x0008,0x0018 (type UI - Unique Identifier)
        ds.SOPInstanceUID = self._ct_sop_instance_uid

        # Study Instance UID tag 0x0020,0x000D (type UI - Unique Identifier)
        # self._dicom_study_instance_uid may be either set in __init__ when creating new object
        #   or set when import a DICOM file
        #   Study Instance UID for structures is the same as Study Instance UID for CTs
        ds.StudyInstanceUID = self._dicom_study_instance_uid

        # Series Instance UID tag 0x0020,0x000E (type UI - Unique Identifier)
        # self._ct_dicom_series_instance_uid may be either set in __init__ when creating new object
        #   or set when import a DICOM file
        #   Series Instance UID for structures might be different than Series Instance UID for CTs
        ds.SeriesInstanceUID = self._ct_dicom_series_instance_uid

        # Study Instance UID tag 0x0020,0x000D (type UI - Unique Identifier)
        ds.FrameOfReferenceUID = '1.2.3'  # !!!!!!!!!
        ds.StudyDate = datetime.datetime.today().strftime('%Y%m%d')
        ds.StudyTime = datetime.datetime.today().strftime('%H%M%S')
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.SamplesPerPixel = 1
        ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']
        ds.Rows = self.dimx
        ds.Columns = self.dimy
        ds.SliceThickness = str(self.slice_distance)
        ds.PixelSpacing = [self.pixel_size, self.pixel_size]

        # Add eclipse friendly IDs
        ds.StudyID = '1'  # Study ID tag 0x0020,0x0010 (type SH - Short String)
        ds.ReferringPhysicianName = 'py^trip'  # Referring Physician's Name tag 0x0008,0x0090 (type PN - Person Name)
        ds.PositionReferenceIndicator = ''  # Position Reference Indicator tag 0x0020,0x1040
        ds.SeriesNumber = '1'  # SeriesNumber tag 0x0020,0x0011 (type IS - Integer String)

        return ds
