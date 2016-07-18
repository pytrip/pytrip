"""
    This file is part of PyTRiP.

    PyTRiP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyTRiP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyTRiP.  If not, see <http://www.gnu.org/licenses/>
"""
import sys


class AptgType:
    CTX = 0
    DOS = 1


class AptgHeader:
    def __init__(self, data, t):
        self.header_type = t
        self.data = data
        self.set_byte_order()

    def get_header(self):
        output_str = "version " + self.version + "\n"
        output_str += "modality " + self.modality + "\n"
        output_str += "created_by " + self.created_by + "\n"
        output_str += "creation_info " + self.creation_info + "\n"
        output_str += "primary_view " + self.primary_view + "\n"
        output_str += "data_type " + self.data_type + "\n"
        output_str += "num_bytes " + str(self.number_bytes) + "\n"
        output_str += "byte_order " + self.byte_order + "\n"
        output_str += "patient_name " + self.patient_name + "\n"
        output_str += "slice_dimension " + str(self.slice_dimension) + "\n"
        output_str += "pixel_size " + str(self.pixel_size) + "\n"
        output_str += "slice_distance " + str(self.slice_distance) + "\n"
        output_str += "slice_number " + str(self.slice_number) + "\n"
        output_str += "xoffset " + str(self.xoffset) + "\n"
        output_str += "dimx " + str(self.dimx) + "\n"
        output_str += "yoffset " + str(self.yoffset) + "\n"
        output_str += "dimy " + str(self.dimy) + "\n"
        output_str += "zoffset " + str(self.zoffset) + "\n"
        output_str += "dimz " + str(self.dimz) + "\n"
        return output_str

    def get_offset(self):
        data = self.data["images"]
        if self.direction == -1:
            idx = len(data) - 1
        else:
            idx = 0
        return data[idx].ImagePositionPatient

    def check_direction(self):
        data = self.data["images"]
        if data[0].ImagePositionPatient[2] < data[1].ImagePositionPatient[2]:
            self.direction = 1
        else:
            self.direction = -1

    def load_dicom(self):
        self.check_direction()
        if self.header_type == AptgType.DOS:
            data = self.data["rtdose"]
            self.slice_number = data.NumberofFrames
        elif self.header_type == AptgType.CTX:
            data = self.data["images"][0]
            self.slice_number = len(self.data["images"])
        self.created_by = ""  # string.replace(_s," ","-")
        self.creation_info = "created by aptg_export;"
        self.primary_view = "transversal"
        self.version = "1.4"
        self.modality = data.Modality
        self.data_type = "integer"
        self.patient_name = data.PatientsName
        self.number_bytes = data.BitsStored / 8
        self.slice_dimension = data.Rows
        self.pixel_size = data.PixelSpacing[0]
        if data.SliceThickness is '':
            self.slice_distance = self.data["images"][0].SliceThickness
        else:
            self.slice_distance = data.SliceThickness
        pos = self.get_offset()
        self.xoffset = int(pos[0]) / self.pixel_size
        self.yoffset = int(pos[1]) / self.pixel_size
        self.zoffset = int(pos[2]) / self.slice_distance
        self.dimx = data.Rows
        self.dimy = data.Columns
        self.dimz = self.slice_number

    def set_byte_order(self):
        endian = sys.byteorder
        if endian == "little":
            self.byte_order = "vms"
            self.byte_format = "<"
        elif endian == "big":
            self.byte_order = "aix"
            self.byte_format = ">"

    def load_voxel(self, path):
        return
