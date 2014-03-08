#! /usr/bin/env python
"""Reads .VDX file from TRiP and Virtuos

bla bla bla
"""

import os, re, sys
import struct
from numpy import *
from pytrip.hed import Header

__author__ = "Niels Bassler"
__version__ = "1.0"
__email__ = "bassler@phys.au.dk"


class Slice(object):

    def __init__(self,i,content,header = None):
        done = False
        self._id = 0 # placeholder for an identifier
        self.header = header
        self.number_of_points = None

        # the vdx slice number is random and not related to anything, dont use.
        # TODO: rename variables to a clear an consistent notation (position, bin, id)
        while not(done):
            if content[i].split()[0] == "slice#":  # FIRST SLICE IS CALLED "1" IN OLD VERSION
                # this is old VDX version. 
                # I do not know what the succeeding coordinates are.
                # looks like simple min max values.
                self.slice = int(content[i].split()[1])
                self.slice_in_frame = (self.slice-1) * self.header.slice_distance # TODO bin 0.5 error? 
                self.thickness = self.header.slice_distance 

            elif re.match("slice", content[i]) is not None: # FIRST SLICE IS CALLED "0" IN 2.0 VERSION
                if content[i].split()[0] == "slice":                
                    self.slice = int(content[i].split()[1]) 
                elif re.match("slice_in_frame", content[i]) is not None:
                    # this is the thing to use. Slice position in mm. 
                    self.slice_in_frame = float(content[i].split()[1])
                    # TODO: consider sorting slices for inc. slice position?
                    # TODO: probably some dict missing as well.

            if re.search("thickness", content[i]) is not None:
                str_list = content[i].split()
                self.thickness = float(str_list[ str_list.index("start_pos")+1 ])
                #self.thickness = float(content[i].split()[1])
            if re.search("reference", content[i]) is not None:
                # TODO: no idea what this is.
                pass
            if re.search("start_pos", content[i]) is not None:
                self.start_pos = float(str_list[ str_list.index("start_pos")+1 ])
    #                self.start_pos = float(content[i].split()[4])
            if re.search("stop_pos", content[i]) is not None:
                self.stop_pos = float(str_list[ str_list.index("start_pos")+1 ])
                #self.stop_pos = float(content[i].split()[6])
            if re.match("number_of_contours", content[i]) is not None:
                self.number_of_contours = float(content[i].split()[1])             
            # TODO: currently only one contour is supported.
            if re.match("internal", content[i]) is not None:
                if (content[i].split()[1]) == "true":
                    self.internal = True
                else: 
                    self.internal = False                
            if re.match("number_of_points", content[i]) is not None:
                number_of_points = int(content[i].split()[1])             
                self.number_of_points = number_of_points
                # init array for data
                self.points = ones((number_of_points,6)) 
                self.vertices = ones((number_of_points,2))
                #j = 0
                for j in range(number_of_points):
                    i += 1
                    self.points[j] = map(float,content[i].split())
                    # print self.point
                    self.vertices[j] = (self.points[j][0], self.points[j][1])

                done = True


            # TRIP VERSION TRIP VERSION TRIP VERSION
            if re.match("points", content[i]) is not None:
                if content[i].split()[0] == "#points":
                    str_list = content[i].split()
                    self.number_of_points = int(str_list[ str_list.index("#points")+1 ])             

                if content[i].split()[0] == "points":
                    str_list = content[i].split()
                    str_list.remove("points") # "points" is first element in this list.
                    temp = array(map(float,str_list))
                    actual_points = len(temp)/2
                    temp2 = temp.reshape(actual_points,2)
                    # TODO: translate into real coordinate system
                    # TODO note that z is missing
                    #
                    # TRiPpos -> Real pos:
                    # It seems that TRiP uses 16 bins per real bin.
                    # thus: conversion factor is pixel_size / 16.0
                    #
                    # Therefore we had to load the header data earlier.
                        #
                    temp2 = self.header.pixel_size/16.0 * temp2
                    #
                    # in order to be compatible with 2.0 virtuos vdx format
                    # the first point is repeated.
                    # TODO: think which way is better:
                    #     1) Ignore last point in 2.0
                    # or  2) repeat 1st point on older version
                    # currently we do 2) for the vertices map.
                    # and nothing for the points[]
                    self.points = ones((actual_points,6))
                    self.vertices = ones((actual_points+1,2)) # see TODO above.
                    self.points[:actual_points,:2] = temp2
                    for j in range(actual_points):
                        self.vertices[j] = (self.points[j][0], self.points[j][1])
                    self.vertices[-1] = (self.points[0][0],self.points[0][1])
                    if self.number_of_points == None:
                        self.number_of_points = actual_points
                    elif self.number_of_points != actual_points:
                        print "VDX: bummer. Error code 120."
                        sys.exit()
                    done = True

    #                print "Completed slice."
    #                print
    #            print "line number i"
            i += 1


class Contour(object):
    def __init__(self,i,content):
        # read until first slice
        done = False
        while not(done):
            #self.slice = [] # Not needed (yet)
            if re.match("reference_frame", content[i]) is not None:
                # dont know what this means.
                pass
                
            if re.match("origin", content[i]) is not None:
                self.origin = [ float(content[i].split()[1]), \
                                    float(content[i].split()[2]),\
                                    float(content[i].split()[3])]
                
            if re.match("point_on_x_axis", content[i]) is not None:
                self.point_on_x_axis = [ float(content[i].split()[1]), \
                                             float(content[i].split()[2]),\
                                             float(content[i].split()[3])]                

            if re.match("point_on_y_axis", content[i]) is not None:
                self.point_on_y_axis = [ float(content[i].split()[1]), \
                                             float(content[i].split()[2]),\
                                             float(content[i].split()[3])]

            if re.match("point_on_z_axis", content[i]) is not None:
                self.point_on_z_axis = [ float(content[i].split()[1]), \
                                             float(content[i].split()[2]),\
                                             float(content[i].split()[3])]
            i += 1

            if re.match("number_of_slices", content[i]) is not None:
                self.nslices = int(content[i].split()[1])
                done = True



class Voi(object):
    def __init__(self,i,content):

        self.slice = [] 
        self.contour = []
        self.prepared = False # index whether mask is prepared
        self.lut_slice = [] # lookup table for slice index <-> slice position

        self.name = None
        self.key = None
        self.type = None

        j = 0
        # TODO not very good, we depend on VOI assignment being terminated by "type" string
        while self.type == None:
            #print i,":",content[i]
            str_list = content[i].split()
            if re.search("voi", content[i]) is not None:
            # this will work even if multiple entries are found on one line.
            # the value is always the next item after the search key
                self.name = str_list[ str_list.index("voi")+1 ]
        #print content[i]
                #print "VDX: found voi", self.name
            if re.search("key", content[i]) is not None:
                self.key = str_list[ str_list.index("key")+1 ]
            if re.search("type", content[i]) is not None:
                self.type = str_list[ str_list.index("type")+1 ]
        #print "VDX: found type", self.type
            i += 1
            j += 1
            if j > 3:
                print "VDX: something bad happend. Code 188."
                sys.exit()
            
        #print "Found voi name: ", self.name

    def bin2slice(self,bin):
        """Supposed to return a slice at position realpos (mm)"""
        return self.lut_slice[bin]

    def slice2bin(self,realpos):
        """Supposed to return a slice at position realpos (mm)"""
        #print "TEST2:", realpos, 
        #print self.lut_slice

        try:
                return self.lut_slice.index(realpos)
        except:
            #if there is no slice in this VOI, return -1
            #print "nope."
            return -1

        # TODO: test if countour exists in slice at all.
        # in fact only slices WITH contours exist in this list.


class Vdx(object):
    def __init__(self, filename=None):
        """ .vdx file handling."""
        print "init vdx"
        self.type  = "VDX"
        self.type_aux = None
        self.name = None
        self.name_aux = None
        self.filename = filename
        self.min = 0
        self.max = 0
        if filename != None:
            self.read(filename)
            
    def __str__(self):
        # print some statistics about the cube
        # len, min, max, and header info.
        return("Not implemented yet.")    
    
    def __add__(self):
        print "Not implemented."
        # TODO: this one could add two contours to one.

    def new(self):
        print "Not implemented."
        # TODO: implement this.

    def read(self,filename):

        """ Read a .vdx file."""
        # TODO: put short and long filenames into object
        FileIsRead = False        
        
        fname_split = os.path.splitext(filename)
        fname = fname_split[0]

        fname_hed = fname + ".hed"
        if os.path.isfile(fname_hed) is False:
            fname_hed = fname +".HED"
            if os.path.isfile(fname_hed) is False:
                raise IOError,  "VDX: Could not find file " \
                      + fname + ".hed or " + fname + ".HED"

        # read auxilliary data from *.hed
        print "VDX: read header file", fname_hed
        self.header = Header(fname_hed)

        #print 'initialized with filename',  filename        
        if os.path.isfile(filename) is False:
            raise IOError,  "VDX: Could not find file " + filename
        else:
            hedinput_file = open( filename , 'r')
            content = hedinput_file.readlines()
            hedinput_file.close()
            data_length = len(content)
            print "VDX: read", data_length,  "lines of data from header file."
            i = 0
            next_voi = 0
            next_contour = 0
            next_slice = 0

        # first check which kind of vdx version we have got.
        self.version = "(none)"
        for i in range(data_length):
                if re.match("vdx_file_version", content[i]) is not None:
                    self.version= content[i].split()[1]

        self.voi = []
        i = 0 # restart

        # VIRTUOS VERSION VIRTUOS VERSION VIRTUOS VERSION
        if self.version == "2.0":
                while i < data_length:
                    #print "parsing line",  i,  content[i]
                    if re.match("all_indices_zero_based", content[i]) is not None:
                        self.all_indices_zero_based = True
                    if re.match("number_of_vois", content[i]) is not None:
                        self.numberofvois = content[i].split()[1]

                    if re.match("voi", content[i]) is not None:
                        # found a VOI
#                        print "Found a VOI."
                        self.voi.append(Voi(i,content))
                        #self.voiname = 
                        next_voi += 1

                    if re.match("contours", content[i]) is not None:
                        # read the contours
#                        print "Found a Contour"
                        self.voi[next_voi-1].contour.append(Contour(i,content))
                        next_contour += 1

                    if re.match("slice", content[i]) is not None:
                        if content[i].split()[0] == "slice":
                            # print "Found a Slice"
                    # TODO check what slice we got, and put it into the right place
                    # No, we cannot do this, because the first slice and its position is
                # not possibly mentioned in the vdx file.
                # therefore this will remain unsorted.
                # instead provide a function which returns a slice at a given position.
                            self.voi[next_voi-1].slice.append(Slice(i,content,self.header))
                # append the position to lookup_table
                            self.voi[next_voi-1].lut_slice.append(self.voi[next_voi-1].slice[-1].slice_in_frame)
                            next_slice += 1
                    i += 1
                FileIsRead = True

        # TRIP VERSION TRIP VERSION TRIP VERION TRIP VERSION
        if self.version != "2.0": # TODO: do less than 2.0 instead of true match
            print "VDX: TRiP style .vdx file."
            while i < data_length:
                #print "parse",  i,  content[i]

                if re.match("voi", content[i]) is not None:
                    #print "VDX: Found a VOI a la TRiP."
                    self.voi.append(Voi(i,content))
                    next_voi += 1

                if re.match("contours", content[i]) is not None:
                    # read the contours
#                        print "Found a Contour"
                    self.voi[next_voi-1].contour.append(Contour(i,content))
                    next_contour += 1
            #print "parse2",  i,  content[i]
                if re.match("slice#", content[i]) is not None:
                    if content[i].split()[0] == "slice#":
                        #print "Found a Slice", next_slice
                # TODO check what slice we got, and put it into the right place
                # No, we cannot do this, because the first slice and its position is
                # not possibly mentioned in the vdx file.
                # therefore this will remain unsorted.
                # instead provide a function which returns a slice at a given position.
                            self.voi[next_voi-1].slice.append(Slice(i,content,self.header))
                # append the position to lookup_table
                # slice_in_frame contains real coordinates, not bins.
                            self.voi[next_voi-1].lut_slice.append(self.voi[next_voi-1].slice[-1].slice_in_frame)
                            next_slice += 1
                i += 1
                FileIsRead = True
            print "Found", next_voi-1, "VOIs"

        self.numberofvois = next_voi-1    
        for i in range(self.numberofvois):
            print "VDX VOI: %20s - Slices: %i" %(self.voi[i].name, len(self.voi[i].slice))

        print "VDX: _fix_vdx"
        
        self._fix_vdx()
        # for convenience
        self.xmin = self.header.xmin
        self.ymin = self.header.ymin
        self.zmin = self.header.zmin
        self.xmax = self.header.xmax
        self.ymax = self.header.ymax
        self.zmax = self.header.zmax
        self.rxmin = self.header.bin2pos(self.xmin)
        self.rymin = self.header.bin2pos(self.ymin)
        self.rzmin = self.header.bin2slice(self.zmin)
        self.rxmax = self.header.bin2pos(self.xmax)
        self.rymax = self.header.bin2pos(self.ymax)
        self.rzmax = self.header.bin2slice(self.zmax)



        print "VDX: done ReadVdx. ------------------------"
        
# TODO Add lookup function for PTV,TARGET, OAR etc.

#    def __del__(self):
#        object.__del__(self)
#    print 'deleted'

#    def __del__(self):
#            object.__del__(self)
#            self.__del__(self)
#            print 'deleted'
    
    def show_version(self):
        print self.version


    def _point_in_polygon(self,polySides,polyX,polyY,x,y):
        """
        //  int    polySides  =  how many corners the polygon has
        //  float  polyX[]    =  horizontal coordinates of corners
        //  float  polyY[]    =  vertical coordinates of corners
        //  float  x, y       =  point to be tested
        """
        oddNodes = False
        j = polySides - 1

    #    print "Point:",x,y
    #    print "There are",polySides,"polynomium edges/sides."

        for i in range(polySides):
            if (polyY[i] < y and polyY[j] >= y or  polyY[j] < y and polyY[i] >= y):
                if (polyX[i]+(y-polyY[i])/(polyY[j]-polyY[i])*(polyX[j]-polyX[i]) < x):
                    oddNodes = not(oddNodes)
            j=i
        return oddNodes





    def import_dicom(self,filename):
        """ Build a VDX object from a dicom object. """

        if _dicom_loaded == False:
            print "pydicom not installed or not available."
            return(None)

        if os.path.isfile(filename) == False:
            print "VDX, dicom_import: cant find file:", filename
        self.header = Header() # empty header object
        #self.
        rt = dicom.read_file(filename)
        if rt.Modality != "RTSTRUCT":
            print "This is not a RTSTRUCT file:", filename, rt.Modality
            return(None)
        #TODO: how to get number of ROIs properly?
        _vois = len(rt.ROIContours) # _vois = number of ROIs/VOIs

        # TODO
        #
        # dont know how to get the contour names?
        # loop over all ROIs
        #     loop over all contours (there may be one or more per slice)
        #         rt.ROIContours[0].Contours[0].NumberofContourPoints
        #         rt.ROIContours[0].Contours[0].ContourData # contains flat array.
        #         update lut. # which is common for all contours
        # fill the header with what is possible.
        #

        # BIG TODO: fix structure of classes:
        # VDX --> VOI[] --> Contours[]
        # and
        # VDX.VOI[].Contours[].GetContourMaskFromSliceNumber(slice_number)
        # or
        # VDX.VOI[].Contours[].GetContourMaskFromSlicePosition(position_in_mm)    
        # or something similar.
        #
        # but obviously, in order to produce this, one needs the header, aargh!
        # GRRR.
        #
    def _fix_vdx(self):
        """ _header is the object from ReadHeader() """
        # this can be moved into ReadVdx
        # get # of slices:
        print "VDX: VdxRead: number of slices in HED:", self.header.slice_number
        print "VDX: First slice is at:", self.header.bin2slice(0), "mm"
        print "VDX: And x y bins are:", self.header.dimx, self.header.dimy

        # build VOI cubes
        self.mask = []

        # TODO alignment checking.
        #print self.voi

        # init empty vois. calculating all takes too long time.    

        for _cv in range(len(self.voi)):
            #print "FIX VDX, voi:", _cv
            # _cv is current voi
                # create empty cube for 1st VOI
            self.mask.append(zeros((self.header.dimx,self.header.dimy,self.header.slice_number),bool)) # inits to False

        print "VDX: done fix_vdx. ------------------------"

    def _calc_voi(self,_cv):
        if self.voi[_cv].prepared == True:
            print "VDX: voi already calculated/prepared"
            return

            print "VDX: Calculate voi number:",_cv, "which is",self.voi[_cv].name
        print "VDX: slice LUT:", self.voi[_cv].lut_slice
        for _cs in range(self.header.slice_number):            
            #_cs is current slice
            
            # lookup index for _cs
            _Rcs = self.header.bin2slice(_cs) # _Rcs holds the real position of slice.
            # does it exist for this VOI?

            _vsi =  self.voi[_cv].slice2bin(_Rcs)
            # _vsi = VDX Slice Index, due to *** internal translation
            #print "VDX:_cs, _Rcs, _vsi:", _cs, _Rcs, _vsi


            if _vsi != -1:
                #print "Slice:", _cs,"at",_Rcs,"mm"
                # ok, we have a contour here.
                _slice = self.voi[_cv].slice[_vsi]

                #print "This slice contains", _slice.number_of_points, "points."

                # TODO obsolete? this is integrated as self.
                # this contains the slice data
                _Rxmin = _slice.points[:,0].min() # these are real positions in mm
                _Rxmax = _slice.points[:,0].max()
                _Rymin = _slice.points[:,1].min()
                _Rymax = _slice.points[:,1].max()
                
                _xmin = self.header.pos2bin(_Rxmin) # these are the translated bin positions
                _xmax = self.header.pos2bin(_Rxmax)
                _ymin = self.header.pos2bin(_Rymin)
                _ymax = self.header.pos2bin(_Rymax)
                x1 = _slice.points[:_slice.number_of_points-1,0]
                y1 = _slice.points[:_slice.number_of_points-1,1]
                x2 = _slice.points[1:,0]
                y2 = _slice.points[1:,1]
                
                gradients = (x2-x1)/(y2-y1)
                
                #print "VDX ranges (bins):", _xmin,_xmax,_ymin,_ymax
                #print "VDX ranges (real):", _Rxmin,_Rxmax,_Rymin,_Rymax
                
                for y in range(_ymin,_ymax):
                    Ry = self.header.bin2pos(y)
                    mask = logical_or(logical_and(y1 < Ry, y2 >= Ry),logical_and(y2 < Ry, y1 >= Ry))
                    xline=self.header.posar2bin((Ry-y1[mask])*gradients[mask]+x1[mask])
                    xline.sort()
                    i = 1
                    result = zeros(size(self.mask[_cv][:,y,_cs]),bool)
                    while i < size(xline):
                        result[xline[i-1]:xline[i]+1]=True
                        i += 2
                        
                    
                    self.mask[_cv][:,y,_cs]=result


                        #print "VDX True at,", _x,_y,_cs

            # TODO: i am not sure if this is right.
        print "VDX: swap axes", _cv
        self.mask[_cv] = swapaxes(self.mask[_cv],0,1)
        self.voi[_cv].prepared = True    # mark that this is prepared.
        print "VDX: done _calc_voi. ------------------------"


    def get_slice_vertices(self, _slice, _voinr):
        # fix index, which starts in zero (heavens forbid!)
        _slice -= 1
        # TODO this is a mess: there are 
        #    i) Absolute slice positions in mm
        #    ii) absolute slice numbers
        #    iii) a slice lookup table.
        _aslice = self.header.bin2slice(_slice)  # translate slice bin to absolute position in mm
        _bslice = self.voi[_voinr].slice2bin(_aslice) # get proper slice number in LUT of current VOI.
        #print "slice bin:", _slice, _aslice, _bslice
        #print self.voi[_voinr].slice[_bslice].vertices
        if _bslice == -1: # no data here.
            return None
        else:
            return (self.voi[_voinr].slice[_bslice].vertices)

    def get_slice(self, _idx, _slice,_voinr):
        # TODO: think very carefully whether this is ok solution.
        # fix index, which starts in zero (heavens forbid!)
        _slice -= 1
        
        self._calc_voi(_voinr)
        _cube = self.mask[_voinr]


        print "x size:" , len(_cube[:,0,0])
        print "y size:" , len(_cube[0,:,0])
        print "z size:" , len(_cube[0,0,:])


#            if type(_slice) == type(1): # int type only supported yet.

        if _idx == "x":
            V = _cube[_slice,:,:]
            print "VDX: slice at x=",_slice,"which is at",self.header.bin2pos(_slice),"mm"
        if _idx == "y":
            V = _cube[:,_slice,:]
            print "VDX: slice at y=",_slice,"which is at",self.header.bin2pos(_slice),"mm"
        if _idx == "z":
            V = _cube[:,:,_slice]
            print "VDX: slice at z=",_slice,"which is at",self.header.bin2slice(_slice),"mm"


        # add half a millimeter to convert from points to bins.
        xmin = self.header.xoffset+0.5
        ymin = self.header.yoffset+0.5
        zmin = self.header.zoffset+0.5
        xmax = xmin + self.header.dimx
        ymax = ymin + self.header.dimy
        zmax = zmin + self.header.dimz

        print "X Y Z ranges: ", xmin,xmax, ymin,ymax, zmin, zmax

        x = arange(xmin,xmax,1) * self.header.pixel_size
        y = arange(ymin,ymax,1) * self.header.pixel_size
        z = (arange(zmin,zmax,1) * self.header.slice_distance) + self.header.bin2slice(0)
#            print "X Y Z arange: ", len(x),len(y),len(z)
        # convert to real coordinates.
        xmin *= self.header.pixel_size
        xmax *= self.header.pixel_size
        ymin *= self.header.pixel_size
        ymax *= self.header.pixel_size
        zmin = self.header.bin2slice(0) # position of first slice
        zmax = zmax * self.header.slice_distance + zmin
        print "Real X Y Z ranges in mm: ", xmin,xmax, ymin,ymax, zmin, zmax

        # plottes der med sz, saa ser vi x langs x og y langs y.
        # plottes der med sy, saa ser vi z langs x og x langs y
        # plottes der med sx, saa ser vi y langs x og z langs y

        if _idx == "x":
            X,Y = meshgrid(z,y)
        if _idx == "y":
            X,Y = meshgrid(z,x)
        if _idx == "z":
            X,Y = meshgrid(x,y)

        print "VDX SHAPE: ", X.shape, Y.shape, V.shape
        print "VDX: done _get_slice. ------------------------"
        return X,Y,V



    
if __name__ == '__main__':    #code to execute if called from command-line

    from optparse import OptionParser

    parser = OptionParser()
#   parser.add_option("-d", "--delta X", dest="dx",
#                 help="Side length of cube to calculate in mm", metavar="int")

    (options, args) = parser.parse_args()

    if len(args) == 0:
        print "Usage:"
        sys.exit()

    filename = args[0]
    V = Vdx(filename)


# from pytrip import *
# V = ReadVdx("testfiles/CBS303000.vdx")
# V.voi[0].slice[1].point
#dir(v.voi[0].slice[1])
#['__class__', '__delattr__', '__dict__', '__doc__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__str__', '__weakref__', 'itnernal', 'number_of_contours', 'number_of_points', 'points', 'slice', 'slice_in_frame', 'start_pos', 'stop_pos', 'thickness', 'vertices']
