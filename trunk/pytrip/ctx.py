#! /usr/bin/env python
"""Reads .CTX file from TRiP

bla bla bla
"""

import os, re, sys, time
import struct
#import string
from pytrip.hed import Header
from numpy import arange,dtype
from pylab import *
import numpy

# DICOM handeling is only optional.
try:
    import dicom
    _dicom_loaded = True
except:
    _dicom_loaded = False


__author__ = "Niels Bassler"
__version__ = "1.0"
__email__ = "bassler@phys.au.dk"



class CtxCube(object):
    """This class reads CTX files."""
    def __init__(self, filename=None):
        """ .ctx file handling."""
        self._id = 0 # placeholder for an identifier
        self.type = "CTX"
        self.type_aux = None
        self.name = None
        self.name_aux = None
        self.filename = filename
        self.min = 0
        self.max = 0

	if filename == None:
	    self.cube = []
	    self.header = []
	    print "init ctx"
	else:
	    self.read(filename)

    def __str__(self):
        # print some statistics about the cube
        # len, min, max, and header info.
        return("Not implemented yet.")    

    def __add__(self):
        print "Not implemented."
        # TODO: this one could add two ctx cubes.

    def new(self):
        print "Not implemented."
        # TODO: implement this.

	
    def read(self, filename):
	    # TODO: fix indentions.
	    """ Read the rst file."""
	    print 'initialized with filename',  filename
	    print "init"
	
	    # test if basename or entire filename was mentioned
	    # fname_split = os.path.splitext(os.path.basename(filename))

	    fname_split = os.path.splitext(filename)
	    if fname_split[1] == ".hed":
		    fname = fname_split[0]
	    elif fname_split[1] == ".ctx":
		    fname = fname_split[0]
	    else:
		    fname = filename
	    
	    # test existence of filename guess
	    fname_hed = fname + ".hed"
            fname_ctx = fname + ".ctx"

	    
            if os.path.isfile(fname_hed) is False:
		    raise IOError,  "Could not find file " + fname_hed

            if os.path.isfile(fname_ctx) is False:
		    raise IOError,  "Could not find file " + fname_ctx

	    self.filename_header = fname_hed
	    self.filename = fname

            # so now we know all our files exist, so lets start the fun
            header = Header(fname_hed)
            #header.read(fname_hed)

            # TODO fix the format. I dont know if we may encounter signed data?
            # here i assume unsigned. see struct.__doc__ to change
            #
            # from the struct manual :
            # "Standard size and alignment are as follows: 
            # no alignment is required for any type
	    # (so you have to use pad bytes); 
            # short is 2 bytes; int and long are 4 bytes. 
            # float and double are 32-bit
	    # and 64-bit IEEE floating point numbers, respectively. "
	    #
            # next read the binary cube
            # check length:
            print "header format_str:",header.format_str
            print "opening ", fname_ctx

            f = open(fname_ctx,"rb")
            a = f.read()
            f.close()
            file_length = len(a)  # file length in bytes.
            cube_size = len(a) / header.num_bytes
            print "--- cube size, dim x,y,z:", \
		  cube_size, header.dimx,header.dimy,header.dimz


            if (header.dimx * header.dimy * header.dimz) != cube_size:
                raise IOError, \
		      "Header size and dose cube size are not consistent."
            print "Read and convert data..."
            #_sformat = header.format_str[0] + \ header.format_str[1] * cube_size
            #_so = struct.Struct(_sformat)
	    scube = zeros(cube_size)
	    print header.format_str
	    so = struct.Struct(header.format_str)
	    print so.unpack(a[0:2])[0]
	    for i in range(cube_size):
            	scube[i] = so.unpack(a[i*header.num_bytes:i*header.num_bytes+header.num_bytes])[0]
            scube.astype(float)
	    print "loaded"
            self.cube = scube.reshape((header.dimx,
                                       header.dimy,
                                       header.dimz),
                                      order='F')

            # TODO this will only go well if x and y are equaliy large. 
            self.cube = swapaxes(self.cube,0,1)
            self.header = header
            self.name = self.header.patient_name
            self.min = self.cube.min()
            self.max = self.cube.max()
            # note that first bin in the cube is at [0,0,0]

            print "CUBEmax CUBEmin: ", self.cube.max(), self.cube.min()
            

    def get_slice(self, _idx, _slice):
        """ Returns a mesh and slice for plotting """
        #print "x bin size:" , len(self.cube[:,0,0])
        #print "y bin size:" , len(self.cube[0,:,0])
        #print "z bin size:" , len(self.cube[0,0,:])

        # indexes start at zero.
        _slice -= 1
        # if type(_slice) == type(1): # int type only supported yet.
        if _idx == "x":
            V = self.cube[_slice,:,:]
        if _idx == "y":
            V = self.cube[:,_slice,:]
        if _idx == "z":
            V = self.cube[:,:,_slice]

        # add half a millimeter to convert from points to bins.
        xmin = self.header.xoffset+0.5
        ymin = self.header.yoffset+0.5
        zmin = self.header.zoffset+0.5
        xmax = xmin + self.header.dimx
        ymax = ymin + self.header.dimy
        zmax = zmin + self.header.dimz

        #print "X Y Z ranges: ", xmin,xmax, ymin,ymax, zmin, zmax
        x = arange(xmin,xmax,1) * self.header.pixel_size
        y = arange(ymin,ymax,1) * self.header.pixel_size
        z = (arange(zmin,zmax,1) * self.header.slice_distance) \
            + self.header.bin2slice(0)
        # print "X Y Z arange: ", len(x),len(y),len(z)
        # convert to real coordinates.
        xmin *= self.header.pixel_size
        xmax *= self.header.pixel_size
        ymin *= self.header.pixel_size
        ymax *= self.header.pixel_size
        zmin = self.header.bin2slice(0) # position of first slice
        zmax = zmax * self.header.slice_distance + zmin
        # print "Real X Y Z ranges in mm: ", xmin,xmax, \
        # ymin,ymax, zmin, zmax
        # plottes der med sz, saa ser vi x langs x og y langs y.
        # plottes der med sy, saa ser vi z langs x og x langs y
        # plottes der med sx, saa ser vi y langs x og z langs y
        if _idx == "x":
            X,Y = meshgrid(z,y)
        if _idx == "y":
            X,Y = meshgrid(z,x)
        if _idx == "z":
            X,Y = meshgrid(x,y)

        # print "CTX SHAPE: ", X.shape, Y.shape, V.shape	    
        return X,Y,V

    def import_dicom(self, filenames):
	    """ Build a CTX cube from a dicom object. """
            if _dicom_loaded == False:
                print "In Soviet Russia, Dicom imports YOU!"
                return(None)

            # most likely all dicom images will have same dimension
            # therefore lets check the first file, and get the basics.

            # define cube to be #slices*xdim*ydim
            # Build common header.
            if os.path.isfile(filenames[0]) == False:
                print "CTX: cant find file:", filenames[0]                
            self.header = Header() # blablah.
            self.header.import_dicom(filenames)
            _cube = []
            for filename in filenames:
                if os.path.isfile(filename):
                    print "CTX: Importing", filename
                    dcm = dicom.read_file(filename)
                    _cube.append(dcm.pixel_array)
                    # TODO: build it properly at numpy level.
                    self.cube = array(_cube)

            self.cube = swapaxes(self.cube,0,2)
            self.cube = rot90(self.cube,3)          
            self.min = self.cube.min()
            self.max = self.cube.max()

    def plot(self, _idx, _slice):
        
        X,Y,V = self.get_slice(_idx,_slice)
        fig = figure()
        ax = fig.add_subplot(111, autoscale_on=False)

        levels = arange(-1010.0,V.max()+10,50.0)
        # plotting bug: for protons, there are voxes with no dose at all.
        # This causes
        # the eps render to produce non-interpretable code.
        # This can be fixed by widening the lowest bin.
        #    levels[0] = -1.0
        #    cbarlist = arange(0,110,10);

        ax.set_xlim(X.min(),X.max())
        ax.set_ylim(Y.min(),Y.max())
        xlabel("ct [mm]") 
        ylabel("ct [mm]")
        ax.set_aspect(1.0)
        grid(True)

        CF = contourf(X,Y,V,levels,cmap=cm.gray,
                      antialiased=True,linewidths=None)
        cb = colorbar(ticks=arange(-1000,3000,200),
                      orientation='vertical')
        cb.set_label('HU')

        majorLocator   = MultipleLocator(1)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator   = MultipleLocator(1.5)
        
        ax.xaxis.set_minor_locator(minorLocator)
        show()
