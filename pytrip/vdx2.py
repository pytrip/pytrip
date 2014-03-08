#! /usr/bin/env python
"""Reads .VDX file from TRiP and Virtuos
"""

import os, re, sys
from error import *
import res.point,string
import struct
from numpy import *

from dicom.dataset import Dataset,FileDataset
from dicom.sequence import Sequence


__author__ = "Niels Bassler and Jakob Toftegaard"
__version__ = "1.0"
__email__ = "bassler@phys.au.dk"
try:
    import dicom
    _dicom_loaded = True
except:
    _dicom_loaded = False
"""
VdxCube is the master class for dealing with vois structures, a vdxcube object contains VoiCube objects which represent a VOI, it
could be ex a lung or the tumor.
The VoiCube contains Slices which correnspons to the CT slices, and the slices contains contour object, which contains the contour data, a slice can contain multiple, since TRiP only support one contour per slice for each voi, it is necessary to merge contour

VdxCube can import both dicom data and TRiP data, and export in the thoose formats.
"""
class VdxCube:
    def __init__(self,content,cube = None):
        self.vois = []
        self.cube = cube
        self.version = "1.2"
    def read_dicom(self,data,structure_ids = None):
        if not data.has_key("rtss"):
            raise InputError, "Input is not a valid rtss structure"
        dcm = data["rtss"]
        self.version = "2.0"
        for i in range(len(dcm.ROIContours)):
            if structure_ids is None or dcm.ROIContours[i].RefdROINumber in structure_ids:
                v = Voi(dcm.RTROIObservations[i].ROIObservationLabel,self.cube)
                v.read_dicom(dcm.RTROIObservations[i],dcm.ROIContours[i])
                self.add_voi(v)
        self.cube.xoffset = 0
        self.cube.yoffset = 0
        self.cube.zoffset = 0
        """shift = min(self.cube.slice_pos)
        for i in range(len(self.cube.slice_pos)):
                self.cube.slice_pos[i] = self.cube.slice_pos[i]-shift"""
    def import_vdx(self,path):
        fp = open(path,"r")
        vdx_data = fp.read().split('\n')
        fp.close()
        self.read_vdx(vdx_data)
    def add_voi(self,voi):
        self.vois.append(voi)
    def get_voi_by_name(self,name):
        for voi in self.vois:
            if voi.name.lower() == name.lower():
                return voi
        raise InputError("Voi doesn't exist")

    def read_vdx(self,content):
        i = 0
        n = len(content)
        header_full = False
        number_of_vois = 0
        while(i < n):
            line = content[i]
            if not header_full:
                if re.match("vdx_file_version",line) is not None:
                    self.version = line.split()[1]
                elif re.match("all_indices_zero_based",line) is not None:
                    self.zero_based = True
                elif re.match("number_of_vois",line) is not None:
                    number_of_vois = int(line.split()[1])
            if re.match("voi",line) is not None:
                v = Voi("")
                if self.version == "1.2":
                    if not line.split()[5] == '0':
                        i = v.read_vdx_old(content,i)
                else:
                    i = v.read_vdx(content,i)
                self.add_voi(v)
                header_full = True
            i+=1

    def concat_contour(self):
        for i in range(len(self.vois)):
            self.vois[i].concat_contour()
    def number_of_vois(self):
        return len(self.vois)
    def write_to_voxel(self,path):
        fp = open(path,"w+")
        fp.write("vdx_file_version %s\n"%self.version)
        fp.write("all_indices_zero_based\n")
        fp.write("number_of_vois %d\n"%self.number_of_vois())
        self.vois = sorted(self.vois, key=lambda voi: voi.type, reverse=True)
        for i in range(len(self.vois)):
            fp.write(self.vois[i].to_voxel_string())
            print
        fp.close()
    def write_to_trip(self,path):
        self.concat_contour()
        self.write_to_voxel(path)
    def create_dicom(self):
        if _dicom_loaded is False:
            raise ModuleNotLoadedError, "Dicom"
        meta = Dataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        meta.MediaStorageSOPInstanceUID = "1.2.3"
        meta.ImplementationClassUID = "1.2.3.4"
        ds = FileDataset("file", {}, file_meta=meta, preamble="\0"*128)
        if self.cube is not None:
            ds.PatientsName = self.patient_name
        else:
            ds.PatientsName = ""
        ds.PatientID = "123456"
        ds.PatientsSex = '0'
        ds.PatientsBirthDate = '19010101'
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.AccessionNumber = ''
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
        ds.SOPInstanceUID = '1.2.3' #!!!!!!!!!!
        ds.StudyInstanceUID = '1.2.3' #!!!!!!!!!!
        ds.SeriesInstanceUID = '1.2.3' #!!!!!!!!!!
        ds.FrameofReferenceUID = '1.2.3' #!!!!!!!!!
        ds.SeriesDate = '19010101' #!!!!!!!!
        ds.ContentDate = '19010101' #!!!!!!
        ds.StudyDate = '19010101' #!!!!!!!
        ds.SeriesTime = '000000' #!!!!!!!!!
        ds.StudyTime = '000000' #!!!!!!!!!!
        ds.ContentTime = '000000' #!!!!!!!!!
        ds.StructureSetLabel = ''
        ds.StructureSetDate = ''
        ds.StructureSetTime = ''
        ds.Modality = 'RTSTRUCT'
        roi_label_list = []
        roi_data_list = []
        roi_structure_roi_list = []

        for i in range(self.number_of_vois()):
            roi_label = self.vois[i].create_dicom_label()
            roi_label.ObservationNumber = str(i+1)
            roi_label.ReferencedROINumber = str(i+1)
            roi_label.RefdROINumber = str(i+1)
            roi_contours = self.vois[i].create_dicom_contour_data(i)
            roi_contours.RefdROINumber = str(i+1)
            roi_contours.ReferencedROINumber = str(i+1)

            roi_structure_roi = self.vois[i].create_dicom_structure_roi()
            roi_structure_roi.ROINumber = str(i+1)

            roi_structure_roi_list.append(roi_structure_roi)
            roi_label_list.append(roi_label)
            roi_data_list.append(roi_contours)
        ds.RTROIObservations = Sequence(roi_label_list)
        ds.ROIContours = Sequence(roi_data_list)
        ds.StructureSetROIs = Sequence(roi_structure_roi_list)
        return ds

    def write_dicom(self,path):
        dcm = self.create_dicom()
        dcm.save_as(os.path.join(path,"rtss.dcm"))
class Voi:
    def __init__(self,name,cube=None):
        self.cube= cube
        self.name = name
        self.is_concated = False
        self.type = 90
        self.slice_z = []
        self.slices = {}
        self.define_colors()
    def concat_to_3d_polygon(self):
        self.concat_contour()
        data = []
        for slice in self.slices:
            data.extend(self.slices[slice].contour[0].contour)
        self.polygon3d = array(data)

    def get_2d_projection_on_basis(self,basis,offset = None):
        a = array(basis[0])
        b = array(basis[1])
        self.concat_contour()
        bas = array([a,b])
        top = []
        bottom = []
        i = 0
        min_value = 10000.0
        max_value = -10000.0
        for slice in self.slices:
            product = dot(self.slices[slice].contour[0].contour,transpose(bas))

            top.append(max(product,key=lambda x: x[0]))
            bottom.append(min(product,key=lambda x: x[0]))
            temp_min = min(product,key=lambda x: x[1])
            temp_max = max(product,key=lambda x: x[1])

            if temp_min[1] < min_value:
                min_value = temp_min[1]
                min_list = product
            if temp_max[1] > max_value:
                max_value = temp_max[1]
                max_list = product
        #Bind structure to ends
        right_max_i = where(max_list[:,0]==max(max_list,key=lambda x: x[0])[0])[0][0]
        right_min_i = where(max_list[:,0]==min(max_list,key=lambda x: x[0])[0])[0][0]
        left_max_i = where(min_list[:,0]==max(min_list,key=lambda x: x[0])[0])[0][0]
        left_min_i = where(min_list[:,0]==min(min_list,key=lambda x: x[0])[0])[0][0]

        left_min = where(min_list[:,1]==min(min_list,key=lambda x: x[1])[1])[0][0]
        right_max= where(max_list[:,1]==max(max_list,key=lambda x: x[1])[1])[0][0]

        if left_max_i < left_min_i:
            temp = left_max_i
            left_max_i = left_min_i
            left_min_i = temp
        if right_max_i < right_min_i:
            temp = right_max_i
            right_max_i = right_min_i
            right_min_i = temp
        if left_min > left_min_i and left_min < left_max_i:
            left = min_list[left_min_i:left_max_i]
        else:
            left = min_list[left_max_i:-1]
            left = append(left,min_list[0:left_min_i],0)

        if right_max > right_min_i and right_max < right_max_i:
            right = max_list[right_min_i:right_max_i]
        else:
            right = max_list[right_max_i:-1]
            right = append(right,max_list[0:right_min_i],0)

        top = sorted(top,key= lambda x: x[1])

        #~ right = reversed(right)
        right = sorted(right,key = lambda x: x[0])
        top.extend(reversed(right))
        bottom = sorted(bottom,key= lambda x: x[1])
        bottom = reversed(bottom)
        top.extend(bottom)
        left = sorted(left,key = lambda x: x[0])
        top.extend(left)


        if offset is not None:
            offset_proj = array([dot(offset,a),dot(offset,b)])
            top = top[:] - offset_proj
        return top


    def define_colors(self):
        self.colors = []
        self.colors.append([0,0,255])
        self.colors.append([0,128,0])
        self.colors.append([0,255,0])
        self.colors.append([255,0,0])
        self.colors.append([0,128,128])
        self.colors.append([255,255,0])
    def calculate_center(self):
        if hasattr(self,"center_pos"):
            return self.center_pos
        self.concat_contour()
        tot_volume = 0
        center_pos = array([0,0,0])
        for key in self.slices:
            center,area = self.slices[key].calculate_center()
            tot_volume += area
            center_pos += area*center
        self.center_pos = center_pos/tot_volume
        return center_pos/tot_volume
    def get_color(self,i):
        return self.colors[i%len(self.colors)]
    def create_dicom_label(self):
        roi_label = Dataset()
        roi_label.ROIObservationLabel = self.name
        roi_label.RTROIInterpretedType = self.get_roi_type_name(self.type)
        return roi_label
    def create_dicom_structure_roi(self):
        roi = Dataset()
        roi.ROIName = self.name
        return roi
    def create_dicom_contour_data(self,i):
        roi_contours = Dataset()
        contours = []
        for k in self.slices:
            contours.extend(self.slices[k].create_dicom_contours())
        roi_contours.Contours = Sequence(contours)
        roi_contours.ROIDisplayColor = self.get_color(i)

        return roi_contours

    def read_vdx_old(self,content,i):
        line = content[i]
        items = line.split()
        self.name = items[1]
        self.type = int(items[3])
        i+=1
        slices = 10000
        while i < len(content):
            line = content[i]
            if re.match("voi",line) is not None:
                break
            if re.match("#TransversalObjects",line) is not None:
                slices = int(line.split()[1])
            i += 1
        print items
        return i-1

    def read_vdx(self,content,i):
        line = content[i]
        self.name = string.join(line.split()[1:],' ')
        number_of_slices = 10000
        i+=1
        while i < len(content):
            line = content[i]
            if re.match("key",line) is not None:
                self.key = line.split()[1]
            elif re.match("type",line) is not None:
                self.type = int(line.split()[1])
            elif re.match("number_of_slices",line) is not None:
                number_of_slices = int(line.split()[1])
            elif re.match("slice",line) is not None:
                s = Slice()
                i = s.read_vdx(content,i)
                key = s.get_position()
                self.slice_z.append(key)
                self.slices[key] = s
            elif re.match("voi",line) is not None:
                print "found voi"
                break
            elif len(self.slices) >= number_of_slices:
                break
            i += 1
        return i-1
    def get_roi_type_number(self,type_name):
        if type_name == 'EXTERNAL':
            return 0
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
    def get_roi_type_name(self,type_id):
        if type_id == 10:
            return "EXTERNAL"
        elif type_id == 2:
            return 'AVOIDANCE'
        elif type_id == 1:
            return 'CTV'
        elif type_id == 0:
            return 'other'
        return ''
    def read_dicom(self,info,data):
        if not data.has_key("Contours"):
            return
        self.name = info.ROIObservationLabel
        type_name = info.RTROIInterpretedType
        self.type = self.get_roi_type_number(typename)
        for i in range(len(data.Contours)):
            key = int(data.Contours[i].ContourData[2])
            if not self.slices.has_key(key):
                self.slices[key] = Slice(self.cube)
                self.slice_z.append(key)
            self.slices[key].add_dicom_contour(data.Contours[i])
    def get_thickness(self):
        if len(self.slice_z) <= 1:
            return 3
        return abs(float(self.slice_z[1])-float(self.slice_z[0]))
    def to_voxel_string(self):
        if len(self.slices) is 0:
            return ""

        out = "\n"
        out += "voi %s\n"%(self.name.replace(" ","_"))
        out += "key empty\n"
        out += "type %s\n"%self.type
        out += "\n"
        out += "contours\n"
        out += "reference_frame\n"
        out += " origin 0.000 0.000 0.000\n"
        out += " point_on_x_axis 1.000 0.000 0.000\n"
        out += " point_on_y_axis 0.000 1.000 0.000\n"
        out += " point_on_z_axis 0.000 0.000 1.000\n"
        out += "number_of_slices %d\n"%self.number_of_slices()
        out += "\n"
        i = 0
        thickness = self.get_thickness()
        for k in self.slice_z:
            sl = self.slices[k]
            pos = sl.get_position()
            out += "slice %d\n"%i
            out += "slice_in_frame %.3f\n"%pos
            out += "thickness %.3f reference start_pos %.3f stop_pos %.3f\n"%(thickness,pos-0.5*thickness,pos+0.5*thickness)
            out += "number_of_contours %d\n"%self.slices[k].number_of_contours()
            out += self.slices[k].to_voxel_string()
            i += 1
        return out
    def get_row_intersections(self,pos):
        slice = self.get_slice_at_pos(pos[2])
        if slice is None:
            return None
        return sort(slice.get_intersections(pos))
    def get_slice_at_pos(self,z):
        thickness = self.get_thickness()/2
        for key in self.slices.keys():
            low = z - thickness
            high = z + thickness
            if (low < key and z > key) or (high > key and z <= key):
                return self.slices[key]
        return None
    def number_of_slices(self):
        return len(self.slices)
    def concat_contour(self):
        if not self.is_concated:
            for k in self.slices.keys():
                self.slices[k].concat_contour()
        self.is_concated  = True
    def get_max_min(self):
        temp_min,temp_max = None,None
        for key in self.slices:
            if temp_min is None:
                temp_min,temp_max = self.slices[key].get_max_min()
            else:
                min1,max1 = self.slices[key].get_max_min()
                temp_min = res.point.min_list(temp_min,min1)
                temp_max = res.point.max_list(temp_max,max1)
        return temp_min,temp_max

class Slice:
    def __init__(self,cube = None):
        self.cube = cube
        self.contour = []
        return
    def add_contour(self,contour):
        self.contour.append(contour)
    def add_dicom_contour(self,dcm):
        offset = []
        offset.append(self.cube.xoffset)
        offset.append(self.cube.yoffset)
        offset.append(min(self.cube.slice_pos))
        self.contour.append(Contour(res.point.array_to_point_array(dcm.ContourData,offset)))
    def get_position(self):
        if len(self.contour) == 0:
            return None
        return self.contour[0].contour[0][2]
    def get_intersections(self,pos):
        intersections = []
        for c in self.contour:
            intersections.extend(res.point.get_x_intersection(pos[1],c.contour))
        return intersections
    def calculate_center(self):
        tot_area = 0
        center_pos = array([0,0,0])
        for contour in self.contour:
            center, area = contour.calculate_center()
            tot_area += area
            center_pos += area*center
        return (center_pos/tot_area,tot_area)

    def read_vdx(self,content,i):
        line = content[i]
        number_of_contours = 0
        i += 1
        while i < len(content):
            line = content[i]
            if re.match("slice_in_frame",line) is not None:
                self.slice_in_frame = float(line.split()[1])
            elif re.match("thickness",line) is not None:
                items = line.split()
                self.thickness = float(items[1])
                if(len(items) == 7):
                    self.start_pos = float(items[4])
                    self.stop_pos = float(items[6])
                else:
                    self.start_pos = float(items[3])
                    self.stop_pos = float(items[5])

            elif re.match("number_of_contours",line) is not None:
                number_of_contours = int(line.split()[1])
            elif re.match("contour",line) is not None:
                c = Contour([])
                i = c.read_vdx(content,i)
                self.add_contour(c)
            elif re.match("slice",line) is not None:
                break
            elif self.contour >= number_of_contours:
                break
            i += 1
        return i-1
    def create_dicom_contours(self):
        contour_list = []
        for i in range(len(self.contour)):
            con = Dataset()
            contour = []
            for p in self.contour[i].contour:
                contour.extend([p[0],p[1],p[2]])
            con.ContourData = contour
            con.ContourGeometricType = 'CLOSED_PLANAR'
            con.NumberofContourPoints = self.contour[i].number_of_points()
            contour_list.append(con)
        return contour_list

    def to_voxel_string(self):
        out = ""
        for i in range(len(self.contour)):
            out += "contour %d\n"%i
            out += "internal false\n"
            out += "number_of_points %d\n"%(self.contour[i].number_of_points()+1)
            out += self.contour[i].to_voxel_string()
            out += "\n"
        return out
    def number_of_contours(self):
        return len(self.contour)
    def concat_contour(self):
        for i in range(len(self.contour)-1,0,-1):
            self.contour[0].push(self.contour[i])
            self.contour.pop(i)
        self.contour[0].concat()
    def get_max_min(self):
        temp_min,temp_max = self.contour[0].get_max_min()
        for i in range(1,len(self.contour)):
            min1,max1 = self.contour[i].get_max_min()
            temp_min = res.point.min_list(temp_min,min1)
            temp_max = res.point.max_list(temp_max,max1)
        return temp_min,temp_max


class Contour:
    def __init__(self,contour,cube=None):
        self.cube = cube
        self.children = []
        self.contour = contour
    def push(self,contour):
        for i in range(len(self.children)):
            if(self.children[i].contains_contour(contour)):
                self.children[i].push(contour)
                return
        self.add_child(contour)
    def calculate_center(self):
        points = self.contour
        points.append(points[-1])
        points = array(points)
        dx_dy = array([points[i+1]-points[i] for i in range(len(points)-1)])
        area = -sum(points[0:len(points)-1,1]*dx_dy[:,0])
        paths = array((dx_dy[:,0]**2+dx_dy[:,1]**2)**0.5)
        total_path = sum(paths)
        center = array([sum(points[0:len(points)-1,0]*paths)/total_path,sum(points[0:len(points)-1:,1]*paths)/total_path,points[0,2]])
        return (center,area)
    def get_max_min(self):
        min_x = amin(array(self.contour)[:,0])
        min_y = amin(array(self.contour)[:,1])
        min_z = amin(array(self.contour)[:,2])

        max_x = amax(array(self.contour)[:,0])
        max_y = amax(array(self.contour)[:,1])
        max_z = amax(array(self.contour)[:,2])
        return ([min_x, min_y,min_z],[max_x,max_y,max_z])

    def to_voxel_string(self):
        out = ""
        for i in range(len(self.contour)):
            out += " %.3f %.3f %.3f %.3f %.3f %.3f\n"%(self.contour[i][0],self.contour[i][1],self.contour[i][2],0,0,0)
        out += " %.3f %.3f %.3f %.3f %.3f %.3f\n"%(self.contour[0][0],self.contour[0][1],self.contour[0][2],0,0,0)
        return out
    def read_vdx(self,content,i):
        set_point = False
        points = 0
        j = 0
        while i < len(content):
            line = content[i]
            if set_point:
                if j >= points-1:
                    break
                con_dat = line.split()
                self.contour.append([float(con_dat[0]), float(con_dat[1]), float(con_dat[2])])
                j += 1
            else:
                if re.match("internal_false",line) is not None:
                    self.internal_false = True
                if re.match("number_of_points",line) is not None:
                    points = int(line.split()[1])
                    set_point = True

            i += 1
        return i-1
    def add_child(self,contour):
        remove_idx = []
        for i in range(len(self.children)):
            if(contour.contains_contour(self.children[i])):
                contour.push(self.children[i])
                remove_idx.append(i)
        remove_idx.sort(reverse=True)
        for i in remove_idx:
            self.children.pop(i)
        self.children.append(contour)
    def number_of_points(self):
        return len(self.contour)
    def has_childs(self):
        if(len(self.children) > 0):
            return True
        return False
    def print_child(self,level):
        for i in range(len(self.children)):
            print level*'\t',
            print self.children[i].contour
            self.children[i].print_child(level+1)

    def contains_contour(self,contour):
        return res.point.point_in_polygon(contour.contour[0][0],contour.contour[0][1],self.contour)
    def concat(self):
        for i in range(len(self.children)):
            self.children[i].concat()
        while(len(self.children) > 1):
            d = -1
            i1 = 0
            i2 = 0
            child = 0
            for i in range(1,len(self.children)):
                i1_temp, i2_temp, d_temp = res.point.short_distance_polygon_idx(self.children[0].contour,self.children[i].contour)
                if(d == -1 or d_temp < d):
                    d = d_temp
                    child = i
            i1_temp, i2_temp, d_temp = res.point.short_distance_polygon_idx(self.children[0].contour,self.contour)
            if d_temp < d:
                self.merge(self.children[0])
                self.children.pop(0)
            else:
                self.children[0].merge(self.children[child])
                self.children.pop(child)
        if(len(self.children) == 1):
            self.merge(self.children[0])
            self.children.pop(0)
    def merge(self,contour):
        if(len(self.contour) == 0):
            self.contour = contour.contour
            return
        i1, i2, d = res.point.short_distance_polygon_idx(self.contour,contour.contour)
        con = []
        for i in range(i1+1):
            con.append(self.contour[i])
        for i in range(i2,len(contour.contour)):
            con.append(contour.contour[i])
        for i in range(i2+1):
            con.append(contour.contour[i])
        for i in range(i1,len(self.contour)):
            con.append(self.contour[i])
        self.contour = con
        return
