from pytrip import dos
from pytrip import ctx
import argparse
import logging
from numpy import arange, meshgrid, ma
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import sys

logger = logging.getLogger(__name__)


def check_compatible(a, b):
    """ Simple comparison of cubes. if X,Y,Z dims are the same, and
    pixel sizes as well, then they are compatible. (Duck typed)
    """
    eps = 1e-5

    if a.dimx != b.dimx:
        logger.error("DIMX does not match: "+str(a.dimx)+" "+str(b.dimx))
        raise Exception("Cubes don't match, check dimx in header.")

    if a.dimy != b.dimy:
        logger.error("DIMY does not match: "+str(a.dimy)+" "+str(b.dimy))
        raise Exception("Cubes don't match, check dimy in header.")

    if a.dimz != b.dimz:
        logger.error("DIMZ does not match: "+str(a.dimz)+" "+str(b.dimz))
        raise Exception("Cubes don't match, check dimz in header.")

    if (a.pixel_size - b.pixel_size) > eps:
        logger.error("Pixel size does not match: "+str(a.pixel_size)+" "+str(b.pixel_size))
        raise Exception("Cubes don't match, check pixel_size in header.")

    if a.slice_dimension != b.slice_dimension:
        logger.error("Slice dimension does not match: "+str(a.slice_dimension)+" "+str(b.slice_dimension))
        raise Exception("Cubes don't match, check slice_dimension in header.")
    return True


def main(args):
    if args.verbosity == 1:
        logger.basicConfig(level=logging.INFO)
    if args.verbosity > 1:
        logger.basicConfig(level=logging.DEBUG)

    logger.info("Dos file: " + args.dos)

    dosbasename = args.dos.split(".")[-2]
    ctxbasename = args.ctx.split(".")[-2]

    d = dos.DosCube()
    fndos = dosbasename + ".dos"
    logger.info("Reading "+fndos)
    d.read(fndos)
    fname_dos = os.path.splitext(args.dos)[0]

    c = ctx.CtxCube()
    fnctx = ctxbasename + ".ctx"
    logger.info("Reading "+fnctx)
    c.read(fnctx)
    logger.info("CTX Cube shape" + str(c.cube.shape))
    logger.info("DOS Cube shape" + str(d.cube.shape))

    # check cube data are compatible
    if d is not None:
        check_compatible(c, d)

    logger.info("Number of slices: " + str(d.dimz))

    xmin = d.xoffset + (0.5*d.pixel_size)  # convert bin to actual position to center of bin
    ymin = d.yoffset + (0.5*d.pixel_size)
    zmin = d.zoffset + (0.5*d.slice_distance)

    xmax = xmin + d.dimx*d.pixel_size
    ymax = ymin + d.dimy*d.pixel_size
    zmax = zmin + d.dimz*d.slice_distance

    dmax = d.cube.max()
    dmin = d.cube.min()

    cmax = c.cube.max()
    cmin = c.cube.min()

    logger.info("Cube first bin pos: {:10.2f} {:10.2f} {:10.2f} [mm]".format(xmin, ymin, zmin))
    logger.info("Cube last bin pos : {:10.2f} {:10.2f} {:10.2f} [mm]".format(xmax, ymax, zmax))
    logger.info("CTX min, max values: {:d} {:d}".format(cmin, cmax))
    logger.info("DOS min, max values: {:d} {:d}".format(dmin, dmax))

    # ctx_levels = arange(-1010.0,cmax+10,50.0)
    # ctx_levels = arange(cmin+10, cmax+10, 50.0)  # alternative ctx colour map
    # dos_levels = arange(0, dmax+50, 5.0)

    x = arange(xmin, xmax, d.pixel_size)
    y = arange(ymin, ymax, d.pixel_size)
    X, Y = meshgrid(x, y)
    Xmax = X.max()
    Xmin = X.min()
    Ymax = Y.max()
    Ymin = Y.min()

    ctx_cb = None
    dos_cb = None

    sstart = args.sstart
    sstop = args.sstop

    if sstart is None:
        sstart = 0
    if sstop is None:
        sstop = d.dimz

    ax = plt.subplot(111, autoscale_on=False)

    # loop over each slice
    for ids in range(sstart, sstop):  # starts at 0

        fnout = fname_dos+"_{:03d}".format(ids)+".png"
        if args.verbosity == 0:
            print("Write slice number: "+str(ids) + "/"+str(d.dimz))
        if args.verbosity > 0:
            logger.info("Write slice number: "+str(ids) + "/"+str(d.dimz) + " to " + fnout)

        ax.cla()
        # ids = 150

        dos_slice = d.cube[ids, :, :]
        # dos_slice *= 0.1  # convert %% to %
        ctx_slice = c.cube[ids, :, :]

        # min/max per slice:
        # dsmax = dos_slice.max()
        # dsmin = dos_slice.min()
        # csmax = ctx_slice.max()
        # csmin = ctx_slice.min()

        plt.xlabel("ct [mm]")
        plt.ylabel("ct [mm]")
        ax.set_aspect(1.0)
        plt.grid(True)

        # extent tells what the x and y coordinates are,
        # so matplotlib can make propper assignment.
        ctx_im = ax.imshow(ctx_slice,
                           cmap=plt.cm.gray,
                           interpolation='bilinear',
                           origin="lower",
                           extent=[Xmin, Xmax,
                                   Ymin, Ymax])
        if args.HUbar:
            if ctx_cb is None:
                ctx_cb = plt.colorbar(ctx_im, ticks=arange(-1000, 3000, 200),
                                      orientation='horizontal')
                ctx_cb.set_label('HU')

        # ------- add dose wash --------

        cmap1 = plt.cm.jet
        cmap1.set_under("k", alpha=0.0)  # seems to be broken! :-( -- Sacrificed goat, now working - David
        cmap1.set_over("k", alpha=0.0)
        cmap1.set_bad("k", alpha=0.0)  # Sacrificial knife here
        tmpdat = ma.masked_where(dos_slice <= dmin, dos_slice)  # Sacrifical goat

        # Do countours
        # dos_im = plt.contourf(X,Y,dos_slice,
        #                      dos_levels,
        #                      cmap=cmap1,
        #                      antialiased=True,linewidths=None)

        dos_im = ax.imshow(tmpdat,
                           interpolation='bilinear',
                           cmap=cmap1,
                           norm=colors.Normalize(vmin=0, vmax=1200, clip=False),
                           alpha=0.7,
                           origin="lower",
                           extent=[Xmin, Xmax,
                                   Ymin, Ymax])

        if dos_cb is None:
            dos_cb = plt.colorbar(dos_im,
                                  # extend='both',
                                  orientation='vertical',
                                  shrink=0.8)
            dos_cb.set_ticks(arange(0, 1300, 200))
            dos_cb.set_label('Relative dose %%')

        ax.set_xlim(Xmax, Xmin)
        ax.set_ylim(Ymax, Ymin)

        # majorLocator = plt.MultipleLocator(1)
        # majorFormatter = plt.FormatStrFormatter('%d')
        minorLocator = plt.MultipleLocator(1.5)

        ax.xaxis.set_minor_locator(minorLocator)
        plt.savefig("foo"+str(ids)+".png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dos", help="doscube to be loaded")
    parser.add_argument("ctx", help="ctxcube to be loaded")  # todo: could be optional
    parser.add_argument("-v", "--verbosity", action='count',
                        help="increase output verbosity", default=0)
    parser.add_argument("-f", "--from", type=int, dest='sstart', metavar='N',
                        help="Output from slice number N", default=0)
    parser.add_argument("-t", "--to", type=int, dest='sstop', metavar='M',
                        help="Output up to slice number M")
    parser.add_argument("-H", "--HUbar", dest='HUbar', default=False, action='store_true',
                        help="Add HU colour bar")
    parser.parse_args()
    args = parser.parse_args()

    sys.exit(main(args))
