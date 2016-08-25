from pytrip import dos
from pytrip import ctx
import argparse
import logging
from numpy import arange,dtype,meshgrid
import matplotlib.pyplot as plt


def check_compatible(a, b):
    """ Simple comparison of cubes. if X,Y,Z dims are the same, and 
    pixel sizes as well, then they are compatible. (Duck typed)
    """
    eps = 1e-5
    
    if a.dimx != b.dimx:
        logging.error("DIMX does not match: "+str(a.dimx)+" "+str(b.dimx))
        raise Exception("Cubes don't match, check dimx in header.")

    if a.dimy != b.dimy:
        logging.error("DIMY does not match: "+str(a.dimy)+" "+str(b.dimy))
        raise Exception("Cubes don't match, check dimy in header.")

    if a.dimz != b.dimz:
        logging.error("DIMZ does not match: "+str(a.dimz)+" "+str(b.dimz))
        raise Exception("Cubes don't match, check dimz in header.")

    if (a.pixel_size - b.pixel_size) > eps:
        logging.error("Pixel size does not match: "+str(a.pixel_size)+" "+str(b.pixel_size))
        raise Exception("Cubes don't match, check pixel_size in header.")

    if a.slice_dimension != b.slice_dimension:
        logging.error("Slice dimension does not match: "+str(a.slice_dimension)+" "+str(b.slice_dimension))
        raise Exception("Cubes don't match, check slice_dimension in header.")
    return True

logging.basicConfig(level=logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument("dos",help="doscube to be loaded")
parser.add_argument("ctx",help="ctxcube to be loaded") # todo: could be optional
parser.parse_args()

args = parser.parse_args()
print(args.dos)

dosbasename = args.dos.split(".")[-2]
ctxbasename = args.ctx.split(".")[-2]
print(dosbasename)
d = dos.DosCube()
d.read(dosbasename+".dos")
#print(dir(d))


c = ctx.CtxCube()
c.read(ctxbasename+".ctx")
print(dir(c))
print(c.cube.shape)
print(d.cube.shape)
print(c.dimx, c.dimy, c.dimz)

# check cube data are compatible
if d != None:
    check_compatible(c,d)

print(d.dimz)

xmin = d.xoffset+(0.5*d.pixel_size) # convert bin to actual position to center of bin
ymin = d.yoffset+(0.5*d.pixel_size)
zmin = d.zoffset+(0.5*d.slice_distance)

xmax = xmin + d.dimx*d.pixel_size
ymax = ymin + d.dimy*d.pixel_size
zmax = zmin + d.dimz*d.slice_distance

x = arange(xmin,xmax,d.pixel_size)
y = arange(ymin,ymax,d.pixel_size)
X,Y = meshgrid(x,y)


# loop over each slice
#for i in range(d.dimz): # starts at 0
#    logging.info("Write slice number: "+str(i))


ids = 150

dos_slice = d.cube[ids,:,:]
ctx_slice = c.cube[ids,:,:]
maxdose = d.cube.max()
maxHU = c.cube.max()


ax = plt.subplot(111, autoscale_on=False)
ctx_levels = arange(-1010.0,maxHU+10,50.0)
dos_levels = arange(0,maxdose+10,50.0)

ax.set_xlim(X.min(),X.max())
ax.set_ylim(Y.min(),Y.max())

plt.xlabel("ct [mm]") 
plt.ylabel("ct [mm]")
ax.set_aspect(1.0)
plt.grid(True)

print(ctx_slice.shape)

CF = plt.contourf(X,Y,ctx_slice,ctx_levels,cmap=plt.cm.gray,
              antialiased=True,linewidths=None)
cb = plt.colorbar(ticks=arange(-1000,3000,200),
              orientation='vertical')
cb.set_label('HU')

majorLocator   = plt.MultipleLocator(1)
majorFormatter = plt.FormatStrFormatter('%d')
minorLocator   = plt.MultipleLocator(1.5)

ax.xaxis.set_minor_locator(minorLocator)
plt.show()

