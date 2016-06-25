"""
    This file is part of PyTRiP.

    libdedx is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    libdedx is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with libdedx.  If not, see <http://www.gnu.org/licenses/>
"""
import os, re, string, sys
from pytrip.error import *
from struct import *
import numpy as np
import array
import pytrip.util

try:
    import dicom
    from dicom.dataset import Dataset, FileDataset

    _dicom_loaded = True
except:
    _dicom_loaded = False


class Cube(object):
    def __init__(self, cube=None):
        if cube is not None:
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
            self.slice_dimension = cube.slice_dimension
            self.pixel_size = cube.pixel_size
            self.slice_distance = cube.slice_distance
            self.slice_number = cube.slice_number
            self.xoffset = cube.xoffset
            self.dimx = cube.dimx
            self.yoffset = cube.yoffset
            self.dimy = cube.dimy
            self.zoffset = cube.zoffset
            self.dimz = cube.dimz
            self.z_table = cube.z_table
            self.slice_pos = cube.slice_pos
            self.set_format_str()
            self.set_number_of_bytes()
            self.cube = np.zeros((self.dimz, self.dimy, self.dimx), dtype=cube.pydata_type)
        else:
            self.header_set = False
            self.version = "2.0"
            self.modality = "CT"
            self.created_by = "pytrip"
            self.creation_info = "create by pytrip"
            self.primary_view = "transversal"  # e.g. transversal
            self.data_type = ""
            self.num_bytes = ""
            self.byte_order = "vms"  # aix or vms
            self.patient_name = ""
            self.slice_dimension = ""  # eg. 256 meaning 256x256 pixels.
            self.pixel_size = ""  # size in mm
            self.slice_distance = ""  # thickness of slice
            self.slice_number = ""  # number of slices in file.
            self.xoffset = 0
            self.dimx = ""  # number of pixels along x (e.g. 256)
            self.yoffset = 0
            self.dimy = ""
            self.zoffset = 0
            self.dimz = ""
            self.slice_pos = []
            self.z_table = False  # list of slice#,pos(mm),thickness(mm),tilt

    def __add__(self, other):
        c = type(self)(self)
        if Cube in other.__class__.__bases__:
            c.cube = other.cube + self.cube
        else:
            c.cube = self.cube + float(other)
        return c

    def __sub__(self, other):
        c = type(self)(self)
        if Cube in other.__class__.__bases__:
            c.cube = self.cube - other.cube
        else:
            c.cube = self.cube - float(other)
        return c

    def __mul__(self, other):
        c = type(self)(self)
        if Cube in other.__class__.__bases__:
            c.cube = other.cube * self.cube
        else:
            t = type(c.cube[0, 0, 0])
            c.cube = np.array(self.cube * float(other), dtype=t)
        return c

    def __div__(self, other):
        c = type(self)(self)
        if Cube in other.__class__.__bases__:
            c.cube = self.cube / other.cube
        else:
            t = type(c.cube[0, 0, 0])
            c.cube = np.array(self.cube / float(other), dtype=t)
        c.cube[np.isnan(c.cube)] = 0  # fix division by zero NaNs
        return c

    def indices_to_pos(self, indices):
        pos = []
        pos.append((indices[0] + 0.5) * self.pixel_size + self.xoffset)
        pos.append((indices[1] + 0.5) * self.pixel_size + self.yoffset)
        pos.append(indices[2] * self.slice_distance + self.zoffset)
        return pos

    def slice_to_z(self, idx):
        return idx * self.slice_distance + self.zoffset

    def pos_to_indices(self, pos):
        indices = []
        indices.append(int(pos[0] / self.pixel_size - self.xoffset / self.pixel_size))
        indices.append(int(pos[1] / self.pixel_size - self.yoffset / self.pixel_size))
        indices.append(int(pos[2] / self.slice_distance - self.zoffset / self.slice_distance))
        return indices

    def get_value_at_indice(self, indices):
        return self.cube[indices[2]][indices[1]][indices[0]]

    def get_value_at_pos(self, pos):
        return self.get_value_at_indice(self.pos_to_indices(pos))

    def create_cube_from_equation(self, equation, center, limits, radial=True):
        eq = util.evaluator(equation);
        data = np.array(np.zeros((self.dimz, self.dimy, self.dimx)))
        x = np.linspace(0.5, self.dimx - 0.5, self.dimx) * self.pixel_size - center[0]
        y = np.linspace(self.dimx - 0.5, 0.5, self.dimx) * self.pixel_size - center[1]
        xv, yv = np.meshgrid(x, y)

    def load_from_structure(self, voi, preset=0, data_type=np.int16):
        data = np.array(np.zeros((self.dimz, self.dimy, self.dimx)), dtype=data_type)
        if preset != 0:
            for i_z in range(self.dimz):
                for i_y in range(self.dimy):
                    intersection = voi.get_row_intersections(self.indices_to_pos([0, i_y, i_z]))
                    if intersection is None:
                        break;
                    if len(intersection) > 0:
                        k = 0
                        for i_x in range(self.dimx):
                            if self.indices_to_pos([i_x, 0, 0])[0] > intersection[k]:
                                k = k + 1
                                if k >= (len(intersection)):
                                    break;
                            if k % 2 == 1:
                                data[i_z][i_y][i_x] = preset
        self.cube = data

    def create_empty_cube(self, value, dimx, dimy, dimz, pixel_size, slice_distance):
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.slice_number = dimz
        self.pixel_size = pixel_size
        self.slice_distance = slice_distance
        self.cube = np.ones((dimz, dimy, dimx), dtype=np.int16) * (value)
        self.slice_dimension = dimz
        self.num_bytes = 2
        self.data_type = "integer"
        self.pydata_type = np.int16

    def override_cube_values(self, voi, value):
        for i_z in range(self.dimz):
            for i_y in range(self.dimy):
                intersection = voi.get_row_intersections(self.indices_to_pos([0, i_y, i_z]))
                if intersection is None:
                    break;
                if len(intersection) > 0:
                    k = 0
                    for i_x in range(self.dimx):
                        if self.indices_to_pos([i_x, 0, 0])[0] > intersection[k]:
                            k = k + 1
                            if k >= (len(intersection)):
                                break;
                        if k % 2 == 1:
                            self.cube[i_z][i_y][i_x] = value

    def set_offset_cube_values(self, voi, value):
        for i_z in range(self.dimz):
            for i_y in range(self.dimy):
                intersection = voi.get_row_intersections(self.indices_to_pos([0, i_y, i_z]))
                if intersection is None:
                    break;
                if len(intersection) > 0:
                    k = 0
                    for i_x in range(self.dimx):
                        if self.indices_to_pos([i_x, 0, 0])[0] > intersection[k]:
                            k = k + 1
                            if k >= (len(intersection)):
                                break;
                        if k % 2 == 1:
                            self.cube[i_z][i_y][i_x] += value

    def write_trip_header(self, path):
        output_str = "version " + self.version + "\n"
        output_str += "modality " + self.modality + "\n"
        output_str += "created_by " + self.created_by + "\n"
        output_str += "creation_info " + self.creation_info + "\n"
        output_str += "primary_view " + self.primary_view + "\n"
        output_str += "data_type " + self.data_type + "\n"
        output_str += "num_bytes " + str(self.num_bytes) + "\n"
        output_str += "byte_order " + self.byte_order + "\n"
        if (self.patient_name == ""):
            self.patient_name = "Anonyme"
        output_str += "patient_name " + self.patient_name + "\n"
        output_str += "slice_dimension " + str(self.slice_dimension) + "\n"
        output_str += "pixel_size " + str(self.pixel_size) + "\n"
        output_str += "slice_distance " + str(self.slice_distance) + "\n"
        output_str += "slice_number " + str(self.slice_number) + "\n"
        # output_str += "xoffset " + str(int(round(self.xoffset/self.pixel_size))) + "\n"
        output_str += "xoffset 0\n"
        output_str += "dimx " + str(self.dimx) + "\n"
        # output_str += "yoffset " + str(int(round(self.yoffset/self.pixel_size))) + "\n"
        output_str += "yoffset 0\n"
        output_str += "dimy " + str(self.dimy) + "\n"
        output_str += "zoffset 0\n"

        """output_str += "zoffset " + str(int(round(self.zoffset/self.slice_distance))) + "\n" """
        output_str += "dimz " + str(self.dimz) + "\n"
        output_str += "z_table no\n"

        """if self.z_table is True:
            output_str += "z_table yes\n"
            output_str += "slice_no  position  thickness  gantry_tilt\n"
            for i in range(len(self.slice_pos)):
                output_str += "  %d\t%.4f\t%.4f\t%.4f\n"%(i+1,self.slice_pos[i],self.slice_distance,0)"""
        with open(path, "w+") as f:
            f.write(output_str)

    def set_byteorder(self, endian=None):
        if endian == None:
            endian = sys.byteorder
        if endian == 'little':
            self.byte_order = "vms"
        elif endian == 'big':
            self.byte_order = "aix"
        else:
            print("HED error: unknown endian:", endian)
            sys.exit(-1)

    def set_format_str(self):
        if (self.byte_order == "vms"):
            self.format_str = "<"
        elif (self.byte_order == "aix"):
            self.format_str = ">"
        self.set_number_of_bytes()

    def set_number_of_bytes(self):
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
        elif self.data_type == "float":
            if self.num_bytes == 4:
                self.format_str += "f"
                self.pydata_type = np.float32
        elif self.data_type == "double":
            if self.num_bytes == 8:
                self.format_str += "d"
                self.pydata_type = np.double
        else:
            print("Format:", self.byte_order, self.data_type, self.num_bytes)
            raise IOError("Unsupported format.")

    def create_dicom_base(self):
        if _dicom_loaded is False:
            raise ModuleNotLoadedError("Dicom")
        if self.header_set is False:
            raise InputError("Header not loaded")
        meta = Dataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        meta.MediaStorageSOPInstanceUID = "1.2.3"
        meta.ImplementationClassUID = "1.2.3.4"
        ds = FileDataset("file", {}, file_meta=meta, preamble="\0" * 128)
        ds.PatientsName = self.patient_name
        ds.PatientID = "123456"
        ds.PatientsSex = '0'
        ds.PatientsBirthDate = '19010101'
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.AccessionNumber = ''
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.file_meta.TransferSyntaxUID = dicom.UID.ImplicitVRLittleEndian
        ds.SOPClassUID = '1.2.3'  # !!!!!!!!
        ds.SOPInstanceUID = '1.2.3'  # !!!!!!!!!!
        ds.StudyInstanceUID = '1.2.3'  # !!!!!!!!!!
        ds.FrameofReferenceUID = '1.2.3'  # !!!!!!!!!
        ds.StudyDate = '19010101'  # !!!!!!!
        ds.StudyTime = '000000'  # !!!!!!!!!!
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.SamplesPerPixel = 1
        ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']
        ds.Rows = self.dimx
        ds.Columns = self.dimy
        ds.SliceThickness = str(self.slice_distance)

        ds.PixelSpacing = [self.pixel_size, self.pixel_size]
        return ds

    def merge(self, cube):
        self.cube = np.maximum(self.cube, cube.cube)

    def merge_zero(self, cube):
        self.cube[self.cube == 0] = cube.cube[self.cube == 0]

    def read_trip_header(self, content):
        i = 0
        self.header_set = True
        content = content.split('\n')
        has_ztable = False
        while i < len(content):
            if re.match("version", content[i]) is not None:
                self.version = content[i].split()[1]
            if re.match("modality", content[i]) is not None:
                self.modality = content[i].split()[1]
            if re.match("created_by", content[i]) is not None:
                self.created_by = string.lstrip(content[i], "created_by ")
                self.created_by = string.rstrip(self.created_by)
            if re.match("creation_info", content[i]) is not None:
                self.creation_info = string.lstrip(content[i], "creation_info ")
                self.creation_info = string.rstrip(self.creation_info)
            if re.match("primary_view", content[i]) is not None:
                self.primary_view = content[i].split()[1]
            if re.match("data_type", content[i]) is not None:
                self.data_type = content[i].split()[1]
            if re.match("num_bytes", content[i]) is not None:
                self.num_bytes = int(content[i].split()[1])
            if re.match("byte_order", content[i]) is not None:
                self.byte_order = content[i].split()[1]
            if re.match("patient_name", content[i]) is not None:
                self.patient_name = content[i].split()[1]
            if re.match("slice_dimension", content[i]) is not None:
                self.slice_dimension = int(content[i].split()[1])
            if re.match("pixel_size", content[i]) is not None:
                self.pixel_size = float(content[i].split()[1])
            if re.match("slice_distance", content[i]) is not None:
                self.slice_distance = float(content[i].split()[1])
            if re.match("slice_number", content[i]) is not None:
                self.slice_number = int(content[i].split()[1])
            if re.match("xoffset", content[i]) is not None:
                self.xoffset = int(content[i].split()[1])
            if re.match("yoffset", content[i]) is not None:
                self.yoffset = int(content[i].split()[1])
            if re.match("zoffset", content[i]) is not None:
                self.zoffset = int(content[i].split()[1])
            if re.match("dimx", content[i]) is not None:
                self.dimx = int(content[i].split()[1])
            if re.match("dimy", content[i]) is not None:
                self.dimy = int(content[i].split()[1])
            if re.match("dimz", content[i]) is not None:
                self.dimz = int(content[i].split()[1])
            if re.match("slice_no", content[i]) is not None:
                self.slice_pos = map(float, range(self.slice_number))
                has_ztable = True
                i += 1
                for j in range(self.slice_number):
                    self.slice_pos[j] = float(content[i].split()[1])
                    i += 1
            i += 1
        self.zoffset = self.zoffset * self.slice_distance
        if has_ztable is not True:
            self.slice_pos = map(float, range(self.slice_number))
            for i in range(self.slice_number):
                self.slice_pos[i] = self.zoffset + i * self.slice_distance
        self.set_format_str()
        self.set_number_of_bytes()

    def read(self, path):
        self.read_trip_data_file(path)

    def read_trip_header_file(self, path):
        f_split = os.path.splitext(path)
        header_file = f_split[0] + ".hed"
        if os.path.isfile(header_file) is False:
            raise IOError("Could not find file " + header_file)
        fp = open(header_file, "r")
        content = fp.read()
        fp.close()
        self.read_trip_header(content)

    def read_trip_data_file(self, path, multiply_by_2=False):
        if self.header_set is False:
            self.read_trip_header_file(path)
        if os.path.isfile(path) is False:
            raise IOError("Could not find file " + path)
        if self.header_set is False:
            raise InputError("Header file not loaded")
        cube = np.fromfile(path, dtype=self.pydata_type)
        if self.byte_order == "aix":
            cube = cube.byteswap()
        if (len(cube) != self.dimx * self.dimy * self.dimz):
            raise IOError("Header size and dose cube size are not consistent.")
        cube = np.reshape(cube, (self.dimz, self.dimy, self.dimx))
        if multiply_by_2 is True:
            cube = cube * 2
        self.cube = cube

    def set_data_type(self, type):
        if (type is np.int8 or type is np.uint8):
            self.data_type = "integer"
            self.num_bytes = 1
        elif (type is np.int16 or type is np.uint16):
            self.data_type = "integer"
            self.num_bytes = 2
        elif (type is np.int32 or type is np.uint32):
            self.data_type = "integer"
            self.num_bytes = 4
        elif (type is np.float):
            self.data_type = "float"
            self.num_bytes = 4
        elif (type is np.double):
            self.data_type = "double"
            self.num_bytes = 8

    def read_dicom_header(self, dcm):
        if _dicom_loaded is False:
            raise ModuleNotLoadedError("Dicom")
        ds = dcm["images"][0]
        self.version = "1.4"
        self.created_by = "pytrip"
        self.creation_info = "created by pytrip;"
        self.primary_view = "transversal"
        self.set_data_type(type(ds.pixel_array[0][0]))
        self.patient_name = ds.PatientsName
        self.slice_dimension = int(ds.Rows)  # should be changed ?
        self.pixel_size = float(ds.PixelSpacing[0])
        self.slice_distance = abs(
            float(dcm["images"][0].ImagePositionPatient[2]) - float(dcm["images"][1].ImagePositionPatient[2]))
        self.slice_number = len(dcm["images"])
        self.xoffset = float(ds.ImagePositionPatient[0])
        self.dimx = int(ds.Rows)
        self.yoffset = float(ds.ImagePositionPatient[1])
        self.dimy = int(ds.Columns)
        self.zoffset = float(ds.ImagePositionPatient[2])
        self.dimz = len(dcm["images"])
        self.z_table = True
        self.set_z_table(dcm)
        self.set_byteorder()
        self.set_format_str()
        self.header_set = True

    def set_z_table(self, dcm):
        self.slice_pos = []
        for i in range(len(dcm["images"])):
            self.slice_pos.append(float(dcm["images"][i].ImagePositionPatient[2]))

    def write_trip_data(self, path):
        cube = np.array(self.cube, dtype=self.pydata_type)
        if self.byte_order == "aix":
            cube = cube.byteswap()
        cube.tofile(path)
        return

        f = open(path, "wb+")
        out = ""
        _format = self.format_str[0] + self.format_str[1] * self.dimx
        i = 0
        for image in self.cube:
            out = ""
            for line in image:
                out += pack(_format, *line)
            f.write(out)
        f.close()
