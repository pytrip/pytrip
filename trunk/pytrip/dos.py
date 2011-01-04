#! /usr/bin/env python
"""Reads .DOS file from TRiP

bla bla bla
"""

import os, re, sys,time
import struct

from pytrip.hed import Header
from pytrip import __file__
from numpy import arange,dtype
from scipy import interpolate
from pylab import *

__author__ = "Niels Bassler"
__copyright__ = "Copyright 2010, Aarhus Particle Therapy Group"
__credits__ = ["Niels Bassler", "David C. Hansen"]
__license__ = "GPL v3"
__version__ = "0.1_svn"
__maintainer__ = "Niels Bassler"
__email__ = "bassler@phys.au.dk"
__status__ = "Development"

class DoseCube(object):
    """This class reads rst files."""
    def __init__(self, filename=None, _type = None):
	""" .dos file handling."""
        self._id = 0 # placeholder for an identifier
        self.type = _type    # either, DOSE, LET, OER, RBE
        self.type_aux = None # optional type defintion ("furusawa..")
        self.name = None     # holding the name, e.g. filename
        self.name_aux = None # optional descriptive string.
        self.max = 0
        self.min = 0
        self.filename = filename
        if filename == None:
            self.cube = []
            self.header = []
            #print "init"
        else:
           self.read(filename)

    def __str__(self):
        # print some statistics about the cube
        # len, min, max, and header info.
        output_str = "type " + str(self.type)+"\n"
        output_str += "type_aux " + str(self.type_aux)+"\n"
        output_str += "name" + str(self.name)+"\n"
        output_str += "name_aux " + str(self.name_aux)+"\n"
        output_str += "filename " + str(self.filename)+"\n"
        output_str += "cube shape: " + str(self.cube.shape)+"\n"
        output_str += "cube size (voxels): " + str(self.cube.size)+"\n"
        output_str += "cube min: "  + str(self.min)+"\n" # min() is slow
        output_str += "cube max: "  + str(self.max)+"\n" # max() is slow
        output_str += "--- Header information ---\n"
        output_str += self.header.__str__()
        return(output_str)

    def __add__(self, other):
        """ sum two cubes or a scalar"""
        c = DoseCube()
        c.header = self.header
        if type(other) == type(self):
            c.cube = self.cube + other.cube # object
        else:
            c.cube = self.cube + float(other) # floats, ints etc.
        return(c)

    def __sub__(self, other):
        # TODO: warn user that header is taken from "self", not "other".
        """ subtract two cubes or a scalar"""
        c = DoseCube()
        c.header = self.header
        if type(other) == type(self):
            c.cube = self.cube - other.cube # object
        else:
            c.cube = self.cube - float(other) # floats, ints etc.
        return(c)

    def __mul__(self, other):
        """ multiply two cubes or a scalar"""
        c = DoseCube()
        c.header = self.header
        if type(other) == type(self):
            c.cube = self.cube * other.cube # object
        else:
            c.cube = self.cube * float(other) # floats, ints etc.
        return(c)

    def __div__(self, other):
        """ divide two cubes or a scalar"""
        c = DoseCube()
        c.header = self.header
        if type(other) == type(self):
            c.cube = self.cube / other.cube # object
        else:
            c.cube = self.cube / float(other) # floats, ints etc.
        c.cube[isnan(c.cube)]=0
        return(c)
    
    def new(self):
        print "DoseCube.new() not implemented yet."

    def read(self,filename):
        fname_split = os.path.splitext(filename)
        if fname_split[1] == ".hed":
            fname = fname_split[0]
        elif fname_split[1] == ".dos":
            fname = fname_split[0]
        else:
            fname = filename
        fname_hed = fname + ".hed"
        fname_dos = fname + ".dos"
        if os.path.isfile(fname_hed) is False:
            raise IOError,  "Could not find file " + fname_hed
        if os.path.isfile(fname_dos) is False:
            raise IOError,  "Could not find file " + fname_dos
        header = Header(fname_hed)

        # TODO fix the format. I dont know if we may encounter signed data?
        # here I assume unsigned. see struct.__doc__ to change
        #
        # from the struct manual :
        # "Standard size and alignment are as follows: 
        # no alignment is required for any type (so you have to use pad bytes); 
        # short is 2 bytes; int and long are 4 bytes. 
        # float and double are 32-bit and 64-bit IEEE floating point numbers, 
        # respectively. "
        
        f = open(fname_dos,"rb")
        a = f.read()
        f.close()
        file_length = len(a)  # file length in bytes.
        cube_size = len(a) / header.num_bytes
        #print "--- cube size, header:dim x,y,z:", \
        #      cube_size, header.dimx,header.dimy,header.dimz


        if (header.dimx * header.dimy * header.dimz) != cube_size:
            raise IOError, "Header size and dose cube size are not consistent."

        _sformat = header.format_str[0] + \
                   header.format_str[1] * cube_size
        _so = struct.Struct(_sformat)
        scube = array(_so.unpack(a))
        scube.astype(float)

        # TODO: verify x-y order at tranpose
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
        # print "CUBEmax CUBEmin: ", self.cube.max(), self.cube.min()

    def BuildOER(self,_dataset = 0):
        """ Returns an OER cube. Dataset is
        0 = OER_barendsen (default)
        1 = OER Furusawa HSG C12
        2 = OER Furusawa V79 C12
        """
        if self.type != "LET":
            print "DOS: error - expected self.type == LET, got", self.type
            return(None)

        if (_dataset > 2):
            print "DOS: Error- only 0,1,2 OER set available. Got:" ,_dataset
        path = os.path.dirname(__file__)
        path_data = (
            os.path.join(path,"data/OER_barendsen.dat"),
            os.path.join(path,"data/OER_furusawa_HSG_C12.dat"),
            os.path.join(path,"data/OER_furusawa_V79_C12.dat"))
        #print "DOS: opening", path_data[_dataset]
        fd = open(path_data[_dataset], 'r')
        lines = fd.readlines()
        fd.close()
        x = [line.split()[0] for line in lines]
        y = [line.split()[1] for line in lines]
        us = interpolate.UnivariateSpline(x, y, s=0.0)
        c = DoseCube()
        _shape = self.cube.shape   # this looks like some kind of bug?
        c.cube = (array(map(us,self.cube))).reshape(_shape)
        c.cube = swapaxes(c.cube,1,2) #  TODO Arrrrgh!
        # c.cube 0 those which are 0 in self.cube
        c.cube[(self.cube<=0.0)] = 0.0
        c.header = self.header
        c.name = c.header.patient_name
        #c.cube = self.cube
        c.type = "OER"
        c.type_aux = path_data[_dataset]
        c.max = c.cube.max()
        c.min = c.cube.min()
        return(c)


    def get_slice(self, _idx, _slice):
        """ Returns a mesh and slice for plotting """
        #print self
        #print "x size:" , len(self.cube[:,0,0])
        #print "y size:" , len(self.cube[0,:,0])
        #print "z size:" , len(self.cube[0,0,:])
        # if type(_slice) == type(1): # int type only supported yet.

        # indexes start at zero.
        _slice -= 1
        
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

        # print "X Y Z ranges: ", xmin,xmax, ymin,ymax, zmin, zmax

        x = arange(xmin,xmax,1) * self.header.pixel_size
        y = arange(ymin,ymax,1) * self.header.pixel_size
        z = (arange(zmin,zmax,1) * self.header.slice_distance) + self.header.bin2slice(0)
        # print "X Y Z arange: ", len(x),len(y),len(z)
        # convert to real coordinates.
        xmin *= self.header.pixel_size
        xmax *= self.header.pixel_size
        ymin *= self.header.pixel_size
        ymax *= self.header.pixel_size
        zmin = self.header.bin2slice(0) # position of first slice
        zmax = zmax * self.header.slice_distance + zmin
        #print "Real X Y Z ranges in mm: ", xmin,xmax, ymin,ymax, zmin, zmax
        if _idx == "x":
            X,Y = meshgrid(z,y)
        if _idx == "y":
            X,Y = meshgrid(z,x)
        if _idx == "z":
            X,Y = meshgrid(x,y)
        #print "DOS SHAPE: ", X.shape, Y.shape, V.shape
        return X,Y,V


    def plot(self, _idx, _slice, _type="dose", filename = None):
        X,Y,V = self.get_slice(_idx,_slice)
        fig = figure()
        ax = fig.add_subplot(111, autoscale_on=False)

        print "min max:", V.min(), V.max()
        if _type == "dose":
            levels  = arange(0,120,1)
            tlevels = arange(0,121,10) #tick levels
        if _type == "let":
            levels  = arange(0.00,200,0.50)
            tlevels = arange(0.00,201,20)
        if _type == "oer":
            levels  = arange(1.00,3.05,0.05)
            tlevels = arange(1.00,3.05,0.5)

        
        # plotting bug: for protons, there are voxes with no dose at all.
        # This causes the eps render to produce non-interpretable code.
        # This can be fixed by widening the lowest bin.
        #    levels[0] = -1.0
        #    cbarlist = arange(0,110,10);

        ax.set_xlim(X.min(),X.max())
        ax.set_ylim(Y.min(),Y.max())
        xlabel("ct [mm]") 
        ylabel("ct [mm]")
        ax.set_aspect(1.0)
        grid(True)

        #CF = contourf(X,Y,V,levels,antialiased=True,linewidths=None)
        #for c in contourf(X,Y,V,levels,antialiased=True).collections:
        #    c.set_linewidth(0.1)
        #cmap1 = cm.get_cmap()
        cmap1 = cm.spectral
        if _type == "oer":
            cmap1 = cm.RdYlBu
        cmap1.set_under(color='k', alpha=0.0)
        cax = ax.imshow(V,
                        interpolation='bilinear',
                        cmap=cmap1,
                        alpha=1,
                        origin="lower",
                        extent=[X.min(),X.max(),
                                Y.min(),Y.max()],
                        vmin=min(levels),
                        vmax=max(levels))

        cb = fig.colorbar(cax, ticks=tlevels, orientation='vertical')
        #cb = colorbar(ticks=tlevels, orientation='vertical')
        #cb = colorbar(ticks=arange(0,600,50), orientation='vertical')
        if _type == "dose":
            cb.set_label('Dose [%]')
        if _type == "let":
            cb.set_label(r'LET [keV/$\mu$m]')
        if _type == "oer":
            cb.set_label('OER')

        majorLocator   = MultipleLocator(1)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator   = MultipleLocator(1.5)

        ax.xaxis.set_minor_locator(minorLocator)


        if filename != None:
            #ax.write_png(filename)
            savefig(filename)
        else:
            show()


    def write(self,filename):
        """ Writes .dos and corresponding .hed file. 
        self.cube: 3d data cube <type 'numpy.ndarray'>
        self.header: header object, as returned by ReadHed()
        filename: string,  suffix will be ignored.
        """
        fname = os.path.splitext(filename)[0] # drop suffix.
        fname_hed = fname + ".hed"
        fname_dos = fname + ".dos"

        self.header.write(fname_hed)

        cube_size = self.header.dimx * self.header.dimy * self.header.dimz
        file_length = cube_size * self.header.num_bytes

        #print "WriteDose, cube size:", cube_size
        #print "WriteDose, file_length:", cube_size

        _sformat = self.header.format_str[0] + \
                   self.header.format_str[1] * cube_size
        _so = struct.Struct(_sformat)

        # C: right index moves fastest
        # fortran : left index moves fastest
        # TODO: some error again. When LET dont swap.
        scube = swapaxes(self.cube,0,1).ravel(True) # True = fortran mode
        #print "length of scube", len(scube)
        #print "length of _sformat", len(_sformat)
        a = _so.pack(*scube ) # * converts list to arguments
        # now write cube
        print "Writing", fname_dos
        f = open(fname_dos,"wb")
        f.write(a)
        f.close()

    
    
if __name__ == '__main__':    #code to execute if called from command-line

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-x", "--slice X", dest="sx",
                  help="Plot slice X", metavar="int")
    parser.add_option("-y", "--slice Y", dest="sy",
                  help="Plot slice Y", metavar="int")
    parser.add_option("-z", "--slice Z", dest="sz",
                  help="Plot slice Z", metavar="int")

    (options, args) = parser.parse_args()

    myfilename = args[0]
    d = ReadDoseCube(myfilename)

    print "x size:" , len(d.cube[:,0,0])
    print "y size:" , len(d.cube[0,:,0])
    print "z size:" , len(d.cube[0,0,:])

    if options.sx != None:
        sx = int(options.sx)
        V = d.cube[sx,:,:]
    elif options.sy != None:
        sy = int(options.sy)
        V = d.cube[:,sy,:]
    elif options.sz != None:
        sz = int(options.sz)
        V = d.cube[:,:,sz]
    else:
        print "Please specify a slice with -x, -y or -z option."
        sys.exit(-1)


    print V.max()
    V = V / V.max() * 100

    # add half a millimeter to convert from points to bins.
    xmin = d.header.xoffset+0.5
    ymin = d.header.yoffset+0.5
    zmin = d.header.zoffset+0.5
    xmax = xmin + d.header.dimx
    ymax = ymin + d.header.dimy
    zmax = zmin + d.header.dimz

    print "X Y Z ranges: ", xmin,xmax, ymin,ymax, zmin, zmax

    x = arange(xmin,xmax,1)
    y = arange(ymin,ymax,1)
    z = arange(zmin,zmax,1)
    print "X Y Z arange: ", len(x),len(y),len(z)

    # plottes der med sz, saa ser vi x langs x og y langs y.
    # plottes der med sy, saa ser vi z langs x og x langs y
    # plottes der med sx, saa ser vi y langs x og z langs y

    fig = figure()
    ax = fig.add_subplot(111, autoscale_on=False)

    if options.sx != None:
        X,Y = meshgrid(y,z)
        xlabel("ct Y [mm]") 
        ylabel("ct SLICE [mm]")
        ax.set_xlim(ymin,ymax)
        ax.set_ylim(zmin,zmax)
    if options.sy != None:
        ylabel("ct X [mm]") 
        xlabel("ct SLICE [mm]")
        X,Y = meshgrid(z,x)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(zmin,zmax)
    if options.sz != None:
        xlabel("ct X [mm]") 
        ylabel("ct Y [mm]")
        X,Y = meshgrid(x,y)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

    levels = arange(0.0,110.0,1.0)
    # plotting bug: for protons, there are voxes with no dose at all. This causes
    # the eps render to produce non-interpretable code.
    # This can be fixed by widening the lowest bin.
    #    levels[0] = -1.0
    #    cbarlist = arange(0,110,10);


    ax.set_aspect(1.0)
    grid(True)

    CF = contourf(V,levels,antialiased=True,linewidths=None)
    # CF = contourf(X,Y,V,levels)

    majorLocator   = MultipleLocator(1)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(1.5)

    ax.xaxis.set_minor_locator(minorLocator)
    show()




# ------ Example --------------
#from pytrip import dos
#from pytrip import 

#from pytrip.hed import Header
#from pytrip.dos import DoseCube

# d = DoseCube("testfiles/CBS303101.dos")
#
# or:
# d = DoseCube()
# d.read("testfiles/CBS303101.dos")
# d.write("foobar.dos")
