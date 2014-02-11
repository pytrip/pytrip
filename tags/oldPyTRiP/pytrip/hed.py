#! /usr/bin/env python
"""Operations on .hed file from TRiP / virtuos

bla bla bla
"""

import os, re, sys
import string
import struct
from numpy import *
try:
    import dicom
    _dicom_loaded = True
except:
    _dicom_loaded = False

__author__ = "Niels Bassler"
__copyright__ = "Copyright 2010, Aarhus Particle Therapy Group"
__credits__ = ["Niels Bassler", "David C. Hansen"]
__license__ = "GPL v3"
__version__ = "0.1_svn"
__maintainer__ = "Niels Bassler"
__email__ = "bassler@phys.au.dk"
__status__ = "Development"


class Header(object):
    'This class read/writes the .hed files.'    
    
    FileIsRead = False
    
    def __init__(self, filename=None):
        """ Init and eventually read the .hed file."""
        
        self._id = 0 # placeholder for an identifier
        
        self.filename = filename
        self.format_str = ""
        
        self.version = ""
        self.modality = ""
        self.created_by = ""
        self.creation_info = ""
        self.primary_view = ""   # e.g. transversal
        self.data_type = ""
        self.num_bytes = ""
        self.byte_order = ""     # aix or vms
        self.patient_name = ""
        self.slice_dimension = ""# eg. 256 meaning 256x256 pixels.
        self.pixel_size = ""     # size in mm
        self.slice_distance = "" # thickness of slice
        self.slice_number = ""   # number of slices in file.
        self.xoffset = ""
        self.dimx = ""           # number of pixels along x (e.g. 256)
        self.yoffset = ""
        self.dimy = ""
        self.zoffset = ""
        self.dimz = ""
        self.z_table = False     # list of slice#,pos(mm),thickness(mm),tilt
        if filename != None:
            self.read(filename)

        self.type = "HED"
        self.type_aux = None
        self.name = None
        self.name_aux = None
        self.filename = filename
    
    def __str__(self):        
        """ Returns string with all variables """
        output_str = "version "+self.version+"\n"
        output_str += "modality "+self.modality+"\n"
        output_str += "created_by " + self.created_by+"\n"
        output_str += "creation_info "+self.creation_info+"\n"
        output_str += "primary_view "+self.primary_view+"\n"
        output_str += "data_type "+self.data_type+"\n"
        output_str += "num_bytes "+str(self.num_bytes)+"\n"
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
        return(output_str)


    def read(self, filename):
        #print "HED: reading header."

        # from version 1.4 this will be filled with data. before we do zeros.
        self.slice_pos = None


        if os.path.isfile(filename) is False:
            raise IOError,  "Could not find file " + filename
        else:
            self.filename = filename
            hedinput_file = open( filename , 'r')
            content = hedinput_file.readlines()
            hedinput_file.close()
            data_length = len(content)
            i = 0
            while i < data_length:
                if re.match("version", content[i]) is not None:
                    self.version= content[i].split()[1]
                if re.match("modality", content[i]) is not None:
                    self.modality = content[i].split()[1]
                if re.match("created_by", content[i]) is not None:
                    self.created_by = string.lstrip(content[i],"created_by ")
                    self.created_by = string.rstrip(self.created_by)
                if re.match("creation_info", content[i]) is not None:
                    self.creation_info = string.lstrip(content[i],
                                                       "creation_info ")
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

                if re.match("z_table", content[i]) is not None:

                    if content[i].split()[1] == "yes":
                        self.z_table = True
                    else:
                        self.z_table = False

                # this does not happen at version 1.2
                if re.match("slice_no", content[i]) is not None:
                    # this should be a list, which provides the index funtion
                    self.slice_pos = map(float,range(self.slice_number))
                    i += 1
                    for j in range(self.slice_number):
                        self.slice_pos[j] = float(content[i].split()[1])
                        i += 1

                i += 1

        # provide zero table for low version numbers 
        if self.slice_pos == None:
            # this should be a list, which provides the index funtion
            self.slice_pos = map(float,range(self.slice_number))
            for j in range(self.slice_number):
                self.slice_pos[j] *= self.slice_distance
            # TODO: may be half bin error here

        FileIsRead = True
        # fix the format. I dont know if we may encounter signed data?
        # here i assume signed. see struct.__doc__ to change
        #
        # from the struct manual :
        # "Standard size and alignment are as follows: 
        # no alignment is required for any type (so you have to use pad bytes); 
        # short is 2 bytes; int and long are 4 bytes. 
        # float and double are 32-bit and 64-bit IEEE floating point numbers, 
        # respectively. "
        if self.byte_order=="aix":
            self.format_str = ">" # big endian
        else:
            self.format_str = "<" # little endian
        if self.data_type == "integer":
            if self.num_bytes==1:
                self.format_str += "b"
                self.pydata_type = int8
            if self.num_bytes==2:
                self.format_str +="h"
                self.pydata_type = int16
            if self.num_bytes==4:
                self.format_str +="i"
                self.pydata_type = int32
        elif self.data_type == "float":                
            if self.num_bytes==4:
                self.format_str +="f"
                self.pydata_type = float
        elif self.data_type == "double":                
            if self.num_bytes==8:
                self.format_str +="d"
                self.pydata_type = double
        else:
            print "Format:", self.byte_order, self.data_type, self.num_bytes
            raise IOError, "Unsupported format."

        # provide boundaries in real coordinate system values.
        # (measured to center of bin)
        self.xmin = self.xoffset+0.5
        self.ymin = self.yoffset+0.5
        self.zmin = self.zoffset+0.5
        self.xmax = self.xmin + self.dimx
        self.ymax = self.ymin + self.dimy
        self.zmax = self.zmin + self.dimz

        # def showVersion(self):
        # print self.version
        
    def pos2bin(self,realpos):
        """ translate pixel XY position in mm to bin """
        # TODO: understand and include offset
        return( int( 0.5 + realpos / self.pixel_size ))

    def posar2bin(self,realposar):
        return( ( 0.5 + realposar / self.pixel_size ).astype(int))
        
    def bin2pos(self,bin):
        """ translate pixel bin to a real position in mm """
        # TODO: understand and include offset
        return( float(bin*self.pixel_size + 0.5 * self.pixel_size))

    def slice2bin(self,realpos):
        """ translate slice position in mm to bin """
        # TODO: understand and include offset
        # guess no rounding here.
        return( int((realpos-self.slice_pos[0]) / self.slice_distance ))  

    def bin2slice(self,bin):
        """ translate pixel bin to a real position in mm """
        # TODO: understand and include offset
        return( float(bin*self.slice_distance + self.slice_pos[0]))

    def set_version(self, version=None):
        """Set header format according to 'version' """
        print "Header: set_version not implemented"

    def set_byteorder(self,endian=None):
        """ Sets endianess of the header.
        'endian' is either 'little' or 'big'.
        If unspecified, the native endianess is used.
        """
        if endian == None:
            endian = sys.byteorder
        if endian == 'little':
            self.byte_order = "vms"
            self.format_str = "<"  + self.format_str[1:]
        elif endian == 'big':
            self.byte_order = "aix"
            self.format_str = ">" + self.format_str[1:]
        else:
            print "HED error: unknown endian:", endian
            sys.exit(-1)

    def import_dicom(self,filenames):
        """ Import header from single dicom file """
        if _dicom_loaded == False:
            print "In Soviet Russia, Dicom imports YOU!"
            return(None)
        if os.path.isfile(filenames[0]) == False:
            print "HED: cant find file:", filenames[0]
            return(None)
        dcm = dicom.read_file(filenames[0])

        self.version = "1.4"
        _s = "%s %s ConvolutionKernel %s" %(dcm.Manufacturer,
                                            dcm.ManufacturersModelName,
                                            dcm.ConvolutionKernel)
        self.created_by = string.replace(_s," ","-")
        self.creation_info = "created by pytrip;"
        self.primary_view = "transversal"
        self.data_type = "integer"
        self.num_bytes = 2      # TODO: guess most are 16 bit ints.
        self.byte_order = "vms" # TODO: make system dependent.
        self.patient_name = dcm.PatientsName
        # it looks like that x and y will always be equal large?
        self.slice_dimension = dcm.Rows # should be changed ?
        self.pixel_size = dcm.PixelSpacing[0]
        self.slice_distance = dcm.SliceThickness
        self.slice_number = len(filenames)
        self.xoffset = 0 # TODO: not ok
        self.dimx = dcm.Rows
        self.yoffset = 0 # TODO: not ok
        self.dimy = dcm.Columns
        self.zoffset = 0 # TODO: not ok
        self.dimz = len(filenames)
        self.z_table = False # TODO: add feature.

        #if self.slice_pos == None:
        # build position table. TODO: may be 0.5*slice_thick wrong.
        self.build_slice_pos_table()

    def build_slice_pos_table(self):
        # this should be a list, which provides the index funtion
        self.slice_pos = map(float,range(self.slice_number))
        for j in range(self.slice_number):
            self.slice_pos[j] *= self.slice_distance
                
    def write(self,filename):
        """ Writes .hed file. 

        cube: 3d data cube

        header: header object, as returned by ReadHed()
        
        """

        f = open(filename, "w")
        f.write(self.__str__() )
        # TODO: handle different versions? This is currently 1.4
        if self.z_table == False:
            f.write("z_table no\n")
        else:
            f.write("z_table yes\n")
            f.write("slice_no  position  thickness  gantry_tilt\n")
            for i in range(self.slice_number):
                f.write("  " + str(i+1) + 
                        "         " + str(self.slice_pos[i]) +
                        "       " + str(self.slice_distance) +
                        "        " + "0.0000" + "\n")
                # TODO: implement gantry tilt.

        f.close()


# usage example:
# from pytrip import *
# h = Header("testfiles/CBS303101.hed")
# print h
# h.set_byteorder("big")
# h.write("Bwhuahhahahaha.hed")
