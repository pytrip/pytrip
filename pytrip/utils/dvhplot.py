#!/usr/bin/env python
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
Script for generating DVH or other volume histograms.
"""
import sys
import logging
import argparse

import pytrip as pt
from pytrip.util import volume_histogram

import matplotlib

logger = logging.getLogger(__name__)


def main(args=sys.argv[1:]):
    """ Main function for dvhplot.py
    """

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("cube", help="Path to input cube. May also be a .dos or dosemlet.dos cube", type=str)
    parser.add_argument("vdx", help="Path to .vdx file holding the structures", type=str)
    parser.add_argument("rois", nargs="?", help="Comma-seperated list for ROIs to be analyzed. If not set, print list.",
                        type=str, default=None)
    parser.add_argument("-d", "--dose", type=float, dest='dose', metavar='dose',
                        help="Taget dose in [Gy]", default=-1.0)
    parser.add_argument("-o", "--output", type=str, dest='output', metavar='filename',
                        help="Don't open GUI, save to filename instead.", default=None)
    parser.add_argument("-l", "--legend", dest='legend', default=False, action='store_true',
                        help="Print legend box")
    parser.add_argument("-v", "--verbosity", action='count', help="increase output verbosity", default=0)
    parser.add_argument('-V', '--version', action='version', version=pt.__version__)
    parsed_args = parser.parse_args(args)

    if parsed_args.verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif parsed_args.verbosity > 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig()

    path_cube = parsed_args.cube
    path_vdx = parsed_args.vdx
    rois_arg = parsed_args.rois
    dose = parsed_args.dose
    legend = parsed_args.legend
    outfile = parsed_args.output

    # there are some cases when this script is run on systems without DISPLAY variable being set
    # in such case matplotlib backend has to be explicitly specified
    # despite the fact interleaving imports with code lines is discouraged, we do it here
    # as it depends on the users options
    if outfile:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = pt.DosCube()
    d.read(path_cube)

    v = pt.VdxCube(d)
    v.read(path_vdx)

    vois = v.get_voi_names()

    if not rois_arg:
        print("Available ROIs:")
        for voi in vois:
            print("'{:s}'".format(voi))
        return

    rois = rois_arg.split(",")

    if d.type == 'DOS':
        if dose < 0.0:
            plt.xlabel("Dose [%]")
        else:
            plt.xlabel("Dose [Gy]")
    elif d.type == 'LET':
        plt.xlabel("LET [keV/um]")
    else:
        plt.xlabel("")  # unknown data cube

    for roi in rois:
        logger.info("Processing ROI '{:s}'...".format(roi))
        voi = v.get_voi_by_name(roi)
        x, y = volume_histogram(d.cube, voi)
        if d.type == 'DOS':
            if dose < 0.0:
                x = x * 0.1
            else:
                x = x * 0.001 * dose

        plt.plot(x, y, label=roi)

    plt.ylabel("Volume [%]")
    plt.grid(True)
    if legend:
        plt.legend()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
