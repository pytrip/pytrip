#
#    Copyright (C) 2010-2017 PyTRiP98 Developers.
#
#    This file is part of PyTRiP98.
#
#    PyTRiP98 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyTRiP98 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyTRiP98.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Generates .png files for each slice found in the given cube.
"""
import argparse
import logging
import os
import sys

from numpy import arange, ma, NINF

import pytrip as pt
from pytrip.util import TRiP98FilePath


def load_data_cube(filename):
    """ Loads either a Dos or LET-cube.

    :params str filname: path to Dos or LET-cube.
    :returns: a DosCube or LETCube object and a str containing the path to the basename.
    """

    if not filename:
        logging.warning("Empty data cube filename")
        return None, None

    logging.info("Reading " + filename)

    d = None
    basename_cube = None

    # try to parse filename, first for LET then DOS cube
    # LET cube file have extension .dosemlet.dos, DOS cube files have extension .dos
    # this means that LET cube file can be interpreted as DOS cube file
    # it is then important to check first for LET cube

    let_path_helper = TRiP98FilePath(filename, pt.LETCube)
    dose_path_helper = TRiP98FilePath(filename, pt.DosCube)

    # check if LET cube type can be determined by presence of suffix in a name (mlet, dosemlet)
    if let_path_helper.suffix is not None:
        cube_cls = let_path_helper.cube_type

    # check if DOS cube type can be determined by presence of suffix in a name (phys, bio etc...)
    elif dose_path_helper.suffix is not None:
        cube_cls = dose_path_helper.cube_type

    # assume dose cube
    else:
        cube_cls = pt.DosCube

    d = cube_cls()  # creating cube object, reading file will be done later

    if d is not None:  # parsing successful, we can proceed to reading the file
        d.read(filename)
        logging.info("Data cube shape" + str(d.cube.shape))
        if isinstance(d, pt.DosCube):
            d *= 0.1  # convert %% to %

        dmax = d.cube.max()
        dmin = d.cube.min()
        logging.info("Data min, max values: {:g} {:g}".format(dmin, dmax))
    else:
        logging.warning("Filename " + filename + " is neither valid DOS nor LET cube")

    basename_cube = d.basename

    return d, basename_cube


def load_ct_cube(filename):
    """ loads the CT cube

    :params str filename: path to filename which must be loaded
    :returns: a CtxCube object and a str containing the path to the basename.
    """

    if not filename:
        logging.warning("Empty CT cube filename")
        return None, None

    logging.info("Reading " + filename)
    c = pt.CtxCube()
    c.read(filename)
    logging.info("CT cube shape" + str(c.cube.shape))

    cmax = c.cube.max()
    cmin = c.cube.min()
    logging.info("CT min, max values: {:d} {:d}".format(cmin, cmax))

    return c, c.basename


def main(args=None):
    """ The main function for cubeslice.py
    """
    if args is None:
        args = sys.argv[1:]

    # there are some cases when this script is run on systems without DISPLAY variable being set
    # in such case matplotlib backend has to be explicitly specified
    # we do it here and not in the top of the file, as interleaving imports with code lines is discouraged
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.ticker import MultipleLocator

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="data cube(dose, LET etc)", type=str, nargs='?')
    parser.add_argument("--ct", help="CT cube", type=str, nargs='?')
    parser.add_argument(
        "-f", "--from", type=int, dest='sstart', metavar='N', help="Output from slice number N", default=1)
    parser.add_argument("-t", "--to", type=int, dest='sstop', metavar='M', help="Output up to slice number M")
    parser.add_argument("-H", "--HUbar", dest='HUbar', default=False, action='store_true', help="Add HU colour bar")
    parser.add_argument("-m", "--max", type=float, dest='csmax', metavar='csmax',
                        help="Maximum colorscale value for plotting data")
    parser.add_argument("-o", "--outputdir", dest='outputdir',
                        help="Directory for output files.", type=str, default=None)
    parser.add_argument('-v', '--verbosity', action='count', help="increase output verbosity", default=0)
    parser.add_argument('-V', '--version', action='version', version=pt.__version__)
    parsed_args = parser.parse_args(args)

    if parsed_args.verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif parsed_args.verbosity > 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig()

    if parsed_args.data is None and parsed_args.ct is None:
        logging.error("Provide location of at least one cube")
        return 2

    # Check if different output path was requested. If yes, then check if it exists.
    if parsed_args.outputdir and not os.path.isdir(parsed_args.outputdir):
        logging.warning("Output directory " + parsed_args.outputdir + " does not exist, creating.")
        os.makedirs(parsed_args.outputdir)

    data_cube, data_basename = load_data_cube(parsed_args.data)

    ct_cube, ct_basename = load_ct_cube(parsed_args.ct)

    # check cube data are compatible (if both cubes present)
    if data_cube is not None and ct_cube is not None and not data_cube.is_compatible(ct_cube):
        logging.error("Cubes don't match")
        logging.info("Cube 1 " + parsed_args.data + " : ")
        logging.info("\tshape {:d} x {:d} x {:d}".format(data_cube.dimx, data_cube.dimy, data_cube.dimz))
        logging.info("\tpixel {:g} [cm], slice {:g} [cm]".format(data_cube.pixel_size, data_cube.slice_distance))
        logging.info("Cube 2 " + parsed_args.ct + " : ")
        logging.info("\tshape {:d} x {:d} x {:d}".format(ct_cube.dimx, ct_cube.dimy, ct_cube.dimz))
        logging.info("\tpixel {:g} [cm], slice {:g} [cm]".format(ct_cube.pixel_size, ct_cube.slice_distance))
        return 2

    cube = None
    cube_basename = None
    if data_cube is not None:  # user provided path to data cube or to data and ct cubes
        cube = data_cube
        cube_basename = data_basename
    elif ct_cube is not None:  # user provided only path to ct cube and not to data cube
        cube = ct_cube
        cube_basename = ct_basename
    else:
        logging.error("Both (data and CT) cubes are empty")
        return 2

    # calculating common frame for printing cubes
    logging.info("Number of slices: " + str(cube.dimz))

    # get actual X and Y positions of bin centers, for two opposite corners, lowest and highest
    # i.e. assuming cube 5x5x5 with pixel_size 1 and slice_distance 1, we will have:
    # xmin_center, ymin_center, zmin_center = 0.5, 0.5, 0
    # xmax_center, ymax_center, zmax_center = 4.5, 4.5, 4
    xmin_center, ymin_center, zmin_center = cube.indices_to_pos([0, 0, 0])
    xmax_center, ymax_center, zmax_center = cube.indices_to_pos([cube.dimx - 1, cube.dimy - 1, cube.dimz - 1])

    logging.info("First bin center: {:10.2f} {:10.2f} {:10.2f} [mm]".format(xmin_center, ymin_center, zmin_center))
    logging.info("Last bin center : {:10.2f} {:10.2f} {:10.2f} [mm]".format(xmax_center, ymax_center, zmax_center))

    # get lowest corner of leftmost lowest bin
    xmin_lowest, ymin_lowest = xmin_center - 0.5 * cube.pixel_size, ymin_center - 0.5 * cube.pixel_size
    # get highest corner of highest rightmost bin
    xmax_highest, ymax_highest = xmax_center + 0.5 * cube.pixel_size, ymax_center + 0.5 * cube.pixel_size

    ct_cb = None
    data_cb = None
    ct_im = None
    data_im = None

    if parsed_args.sstart <= 0:
        logging.error("Invalid slice number {:d}. First slice must be 1 or greater.".format(parsed_args.sstart))
        return 1

    # if user hasn't provided number of first slice, then argument parsing method will set it to zero
    slice_start = parsed_args.sstart

    # user hasn't provided number of last slice, we assume then that the last one has number cube.dimz
    slice_stop = parsed_args.sstop
    if slice_stop is None:
        slice_stop = cube.dimz

    # user hasn't provided maximum limit of colorscale, we assume then maximum value of data cube, if present
    # we clip data to the maximum value of colorscale
    data_colorscale_max = parsed_args.csmax
    if data_colorscale_max is None and data_cube is not None:
        data_colorscale_max = cube.cube.max()
    data_cube.cube.clip(NINF, data_colorscale_max, data_cube.cube)

    # Prepare figure and subplot (axis), they will stay the same during the loop
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False)

    # Set axis
    ax.set_xlabel("[mm]")
    ax.set_ylabel("[mm]")

    ax.set_aspect(1.0)
    for ax in fig.axes:
        ax.grid(True)

    ax.set_xlim(xmin_lowest, xmax_highest)
    ax.set_ylim(ymin_lowest, ymax_highest)

    _minor_locator = MultipleLocator(cube.pixel_size)
    ax.xaxis.set_minor_locator(_minor_locator)
    ax.yaxis.set_minor_locator(_minor_locator)

    text_slice = ax.text(0.0, 1.01, "", size=8, transform=ax.transAxes)

    # loop over each slice
    for slice_id in range(slice_start, slice_stop + 1):

        output_filename = "{:s}_{:03d}.png".format(cube_basename, slice_id)

        slice_str = str(slice_id) + "/" + str(cube.dimz)
        if parsed_args.outputdir is not None:
            # use os.path.join() in order to add slash if it is missing.
            output_filename = os.path.join(parsed_args.outputdir, os.path.basename(output_filename))
        if parsed_args.verbosity == 0:
            print("Write slice number: " + slice_str)
        elif parsed_args.verbosity > 0:
            logging.info("Write slice number: " + slice_str + " to " + output_filename)

        slice_str = "Slice #: {:3d}/{:3d}\nSlice position: {:6.2f} mm".format(slice_id,
                                                                              cube.dimz,
                                                                              cube.slice_to_z(slice_id))
        text_slice.set_text(slice_str)

        if ct_cube is not None:
            ct_slice = ct_cube.cube[slice_id - 1, :, :]  # extract 2D slice from 3D cube

            # remove CT image from the current plot (if present) and replace later with new data
            if ct_im is not None:
                ct_im.remove()

            ct_im = ax.imshow(
                ct_slice,
                cmap=plt.cm.gray,
                interpolation='nearest',  # each pixel will get colour of nearest neighbour, useful when displaying
                #  dataset of lower resolution than the output image
                #  see http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
                origin="lower",  # ['upper' | 'lower']:
                # Place the [0,0] index of the array in the upper left or lower left corner of the axes.
                extent=[xmin_lowest, xmax_highest, ymin_lowest, ymax_highest]  # scalars (left, right, bottom, top) :
                # the location, in data-coordinates, of the lower-left and upper-right corners
            )

            # optionally add HU bar
            if parsed_args.HUbar and ct_cb is None:
                ct_cb = fig.colorbar(ct_im, ax=ax, ticks=arange(-1000, 3000, 200), orientation='horizontal')
                ct_cb.set_label('HU')

        if data_cube is not None:
            data_slice = data_cube.cube[slice_id - 1, :, :]  # extract 2D slice from 3D cube

            # remove data cube image from the current plot (if present) and replace later with new data
            if data_im is not None:
                data_im.remove()

            cmap1 = plt.cm.jet
            cmap1.set_under("k", alpha=0.0)  # seems to be broken! :-( -- Sacrificed goat, now working - David
            cmap1.set_over("k", alpha=0.0)
            cmap1.set_bad("k", alpha=0.0)  # Sacrificial knife here
            dmin = data_cube.cube.min()
            dmax = data_cube.cube.max() * 1.1
            if data_colorscale_max is not None and data_cube is not None:
                dmax = data_colorscale_max * 1.1
            tmpdat = ma.masked_where(data_slice <= dmin, data_slice)  # Sacrificial goat

            # plot new data cube
            data_im = ax.imshow(
                tmpdat,
                cmap=cmap1,
                norm=colors.Normalize(vmin=0, vmax=dmax, clip=False),
                alpha=0.7,
                interpolation='nearest',  # each pixel will get colour of nearest neighbour, useful when displaying
                #  dataset of lower resolution than the output image
                #  see http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
                origin="lower",  # ['upper' | 'lower']:
                # Place the [0,0] index of the array in the upper left or lower left corner of the axes.
                extent=[xmin_lowest, xmax_highest, ymin_lowest, ymax_highest]  # scalars (left, right, bottom, top) :
                # the location, in data-coordinates, of the lower-left and upper-right corners
            )

            if data_cb is None:
                data_cb = fig.colorbar(
                    data_im,
                    orientation='vertical',
                    shrink=0.8)
                if isinstance(data_cube, pt.LETCube):
                    data_cb.set_label("LET [keV/um]")
                elif isinstance(data_cube, pt.DosCube):
                    data_cb.set_label("Relative dose [%]")

        # by default savefig will produce 800x600 resolution, setting dpi=200 is increasing it to 1600x1200
        fig.savefig(output_filename, dpi=200)

    logging.info("Done")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
