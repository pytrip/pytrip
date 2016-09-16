import os
import sys
import argparse
import logging

from matplotlib.ticker import MultipleLocator
from numpy import arange, meshgrid, ma

from pytrip import dos, ctx, let

logger = logging.getLogger(__name__)


def load_data_cube(filename):

    if not filename:
        logger.warn("Empty data cube filename")
        return None, None

    logger.info("Reading " + filename)

    d = None
    basename_cube = None

    # try to load LET cube
    data_header, _ = let.LETCube.parse_path(filename)
    if data_header is not None:
        basename_cube = os.path.splitext(data_header)[0]
        d = let.LETCube()

    # try to load DOS cube
    data_header, _ = dos.DosCube.parse_path(filename)
    if d is None and data_header is not None:
        basename_cube = os.path.splitext(data_header)[0]
        d = dos.DosCube()

    if d is not None:
        d.read(filename)
        logger.info("Data cube shape" + str(d.cube.shape))
        if isinstance(d, dos.DosCube):
            d *= 0.1  # convert %% to %

        dmax = d.cube.max()
        dmin = d.cube.min()
        logger.info("Data min, max values: {:g} {:g}".format(dmin, dmax))

    if d is None:
        logger.warn("Filename " + filename + " is neither valid DOS neither LET cube")

    return d, basename_cube


def load_ct_cube(filename):
    # load CT

    if not filename:
        logger.warn("Empty CT cube filename")
        return None, None

    logger.info("Reading " + filename)
    ctx_header, _ = ctx.CtxCube.parse_path(filename)

    if ctx_header is None:
        logger.warn("Path " + filename + " doesn't seem to point to proper CT cube")
        return None, None

    basename_cube = os.path.splitext(ctx_header)[0]
    c = ctx.CtxCube()
    c.read(filename)
    logger.info("CT cube shape" + str(c.cube.shape))

    cmax = c.cube.max()
    cmin = c.cube.min()
    logger.info("CT min, max values: {:d} {:d}".format(cmin, cmax))

    return c, basename_cube


def main(args=sys.argv[1:]):

    # there are some cases when this script is run on systems without DISPLAY variable being set
    # in such case matplotlib backend has to be explicitly specified
    # we do it here and not in the top of the file, as inteleaving imports with code lines is discouraged
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="data cube(dos, let etc)", type=str, nargs='?')
    parser.add_argument("--ct", help="CT cube", type=str, nargs='?')
    parser.add_argument("-v", "--verbosity", action='count', help="increase output verbosity", default=0)
    parser.add_argument(
        "-f", "--from", type=int, dest='sstart', metavar='N', help="Output from slice number N", default=0)
    parser.add_argument("-t", "--to", type=int, dest='sstop', metavar='M', help="Output up to slice number M")
    parser.add_argument("-H", "--HUbar", dest='HUbar', default=False, action='store_true', help="Add HU colour bar")
    parser.add_argument("-o", "--outputdir", dest='outputdir',
                        help="Write resulting files to this directory.", type=str, default=None)
    args = parser.parse_args(args)

    if args.verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    if args.verbosity > 1:
        logging.basicConfig(level=logging.DEBUG)

    if args.data is None and args.ct is None:
        logger.error("Provide location of at least one cube")
        return 2

    # Check if different output path was requested. If yes, then check if it exists.
    if args.outputdir is not None:
        if os.path.isdir(args.outputdir) is False:
            logger.error("Output directory " + args.outputdir + "does not exist.")
            return 1

    data_cube, data_basename = load_data_cube(args.data)

    ct_cube, ct_basename = load_ct_cube(args.ct)

    # check cube data are compatible (if both cubes present)
    if data_cube is not None and ct_cube is not None:
        if not data_cube.is_compatible(ct_cube):
            logger.error("Cubes don't match")
            logger.info("Cube 1 " + args.dos + " shape {:d} x {:d} x {:d}".format(data_cube.dimx, data_cube.dimy,
                                                                                  data_cube.dimz))
            logger.info("Cube 1 " + args.dos + " pixel {:g} [cm], slice {:g} [cm]".format(data_cube.pixel_size,
                                                                                          data_cube.slice_distance))
            logger.info("Cube 2 " + args.ctx + " shape {:d} x {:d} x {:d}".format(ct_cube.dimx, ct_cube.dimy,
                                                                                  ct_cube.dimz))
            logger.info("Cube 2 " + args.ctx + " pixel {:g} [cm], slice {:g} [cm]".format(ct_cube.pixel_size,
                                                                                          ct_cube.slice_distance))
            return 2

    cube = None
    cube_basename = None
    if data_cube is not None:
        cube = data_cube
        cube_basename = data_basename
    elif ct_cube is not None:
        cube = ct_cube
        cube_basename = ct_basename
    else:
        logger.error("Both (data and CT) cubes are empty")
        return 2

    # calculating common frame for printing cubes
    logger.info("Number of slices: " + str(cube.dimz))

    # convert bin to actual position to center of bin
    xmin = cube.xoffset + 0.5 * cube.pixel_size
    ymin = cube.yoffset + 0.5 * cube.pixel_size
    zmin = cube.zoffset + 0.5 * cube.slice_distance

    xmax = xmin + cube.dimx * cube.pixel_size
    ymax = ymin + cube.dimy * cube.pixel_size
    zmax = zmin + cube.dimz * cube.slice_distance

    logger.info("First bin pos: {:10.2f} {:10.2f} {:10.2f} [mm]".format(xmin, ymin, zmin))
    logger.info("Last bin pos : {:10.2f} {:10.2f} {:10.2f} [mm]".format(xmax, ymax, zmax))

    x = arange(xmin, xmax, cube.pixel_size)
    y = arange(ymin, ymax, cube.pixel_size)
    x_grid, y_grid = meshgrid(x, y)
    x_max = x_grid.max()
    x_min = x_grid.min()
    y_max = y_grid.max()
    y_min = y_grid.min()

    ct_cb = None
    data_cb = None
    ct_im = None
    data_im = None

    slice_start = args.sstart
    slice_stop = args.sstop

    if slice_stop is None:
        slice_stop = cube.dimz

    # Prepare figure and subplot (axis)
    # They will stay the same during the loop
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False)

    # Set axis
    ax.set_xlabel("[mm]")
    ax.set_ylabel("[mm]")

    ax.set_aspect(1.0)
    for ax in fig.axes:
        ax.grid(True)

    ax.set_xlim(x_max, x_min)
    ax.set_ylim(y_max, y_min)

    minorLocator = MultipleLocator(1.5)
    ax.xaxis.set_minor_locator(minorLocator)

    # loop over each slice
    for ids in range(slice_start, slice_stop):  # starts at 0

        output_filename = cube_basename + "_{:03d}".format(ids) + ".png"
        if args.outputdir is not None:
            # use os.path.join() in order to add slash if it is missing.
            output_filename = os.path.join(args.outputdir, os.path.basename(output_filename))
        if args.verbosity == 0:
            print("Write slice number: " + str(ids) + "/" + str(cube.dimz))
        if args.verbosity > 0:
            logger.info("Write slice number: " + str(ids) + "/" + str(cube.dimz) + " to " + output_filename)

        if ct_cube is not None:
            ct_slice = ct_cube.cube[ids, :, :]

            # remove CT image from the current plot (if present) and replace later with new data
            if ct_im is not None:
                ct_im.remove()

            ct_im = ax.imshow(
                ct_slice,
                cmap=plt.cm.gray,
                interpolation='bilinear',
                origin="lower",
                extent=[x_min, x_max, y_min, y_max])  # extent tells what the x and y coordinates are

            # optionally add HU bar
            if args.HUbar and ct_cb is None:
                ct_cb = fig.colorbar(ct_im, ax=ax, ticks=arange(-1000, 3000, 200), orientation='horizontal')
                ct_cb.set_label('HU')

        if data_cube is not None:
            # Extract the slice
            data_slice = data_cube.cube[ids, :, :]

            # remove data cube image from the current plot (if present) and replace later with new data
            if data_im is not None:
                data_im.remove()

            cmap1 = plt.cm.jet
            cmap1.set_under("k", alpha=0.0)  # seems to be broken! :-( -- Sacrificed goat, now working - David
            cmap1.set_over("k", alpha=0.0)
            cmap1.set_bad("k", alpha=0.0)  # Sacrificial knife here
            dmin = data_cube.cube.min()
            dmax = data_cube.cube.max()
            tmpdat = ma.masked_where(data_slice <= dmin, data_slice)  # Sacrifical goat

            # plot new data cube
            data_im = ax.imshow(
                tmpdat,
                interpolation='bilinear',
                cmap=cmap1,
                norm=colors.Normalize(
                    vmin=0, vmax=dmax * 1.1, clip=False),
                alpha=0.7,
                origin="lower",
                extent=[x_min, x_max, y_min, y_max])

            if data_cb is None:
                data_cb = fig.colorbar(
                    data_im,
                    # extend='both',
                    orientation='vertical',
                    shrink=0.8)
                if isinstance(data_cube, let.LETCube):
                    data_cb.set_label("LET [keV/um]")
                elif isinstance(data_cube, dos.DosCube):
                    data_cb.set_label("Relative dose [%]")

        fig.savefig(output_filename)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
