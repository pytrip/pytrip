#!/usr/bin/env python
#
#    Copyright (C) 2020 PyTRiP98 Developers.
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
import os
import sys
import logging
import argparse
import numpy as np
import pytrip as pt

logger = logging.getLogger(__name__)


class DDD(object):
    """This class handles DDD files."""
    # TODO: this class could go into main pytrip/ as well.

    def __init__(self):
        """ setup of object """
        self.path = None
        self.filename = None  # filename, without directory
        self.filetype = "ddd"
        self.fileversion = "19980520"  # default
        self.filedate = None  # format like "Sat Sep 12 14:48:56 2020"
        self.projectile = ""  # "12C"
        self.material = ""  # H2O
        self.composition = ""  # H2O
        self.density = 1.0
        self.energy = 0.0  # in MeV/u (NOT per nucleon)
        self.ngauss = 0  # number of gaussian fits found in this file
        self.ndata = 0  # number of data points in this set
        self.data = None

    def read(self, fn):
        """ Reads the filename set when constructing the ReadGd class
        """
        if os.path.isfile(fn) is False:
            raise IOError("Could not find file " + fn)

        self.path = fn
        self.filename = os.path.basename(fn)

        with open(fn) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            _l = line.strip().split(' ', 1)
            if len(_l) == 2:
                key, value = _l
            if key == "!filetype":
                self.filetype = value
            if key == "!fileversion":
                self.fileversion = value
            if key == "!filedate":
                self.filedate = value
            if key == "!projectile":
                self.projectile = value
            if key == "!material":
                self.material = value
            if key == "!composition":
                self.composition = value
            if key == "!density":
                self.density = float(value)
            if key == "!energy":
                self.energy = float(value)

        self.data = np.loadtxt(fn, comments=("#", "!"))

        rows, cols = self.data.shape

        if rows < 1:
            raise IOError("Invalid number of data points in DDD file: " + fn)
        self.ndata = rows

        if cols < 2:
            raise IOError("Invalid number of columns in DDD file: " + fn)
        self.ngauss = cols - 2  # number of gaussian fits donw

    def savefig(self, type="png"):
        """
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # add suffix
        outfile = self.filename + "." + type

        fig, (p1, p2) = plt.subplots(2, 1)

        p1.set(ylabel="dE/dz [MeV/(g/cm**2)]", title="{:s} - {:.3f} MeV/u".format(self.projectile, self.energy))
        # p1.set(xlabel="z [g/cm**2]", ylabel="dE/dz [MeV/(g/cm**2)]")
        p1.grid()
        p1.plot(self.data[:, 0], self.data[:, 1])

        if self.ngauss > 0:
            p2.set(xlabel="z [g/cm**2]", ylabel="FWHM [g/cm**2]")
            # p2.set_ylim([0, 2])
            p2.set_yscale('log')
            p2.grid(which="both")
            p2.tick_params(axis='both', which='minor', labelsize=6)
            # p2.yaxis.set_label_position("right")
            # p2.yaxis.tick_right()
            from matplotlib.ticker import FormatStrFormatter
            p2.yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
            for n in range(self.ngauss):
                p2.plot(self.data[:, 0], self.data[:, n + 2])

        plt.savefig(outfile)
        plt.close()

    @classmethod
    def gauss(x, fwhm):
        """ For future use
        """
        mu = 0
        s = fwhm / 2.35482
        return (1 / (s * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * s**2)))


def main(args=sys.argv[1:]):
    """
    Hint:
    A series of .png files can be coverted to an animated gif using
    $ convert -delay 0.1 -loop 0 *.png animated.gif
    """

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("ddd_files", help="path to input DDD file", type=argparse.FileType('r'), nargs='+')
    parser.add_argument('-v', '--verbosity', action='count', help="increase output verbosity", default=0)
    parser.add_argument('-V', '--version', action='version', version=pt.__version__)
    parsed_args = parser.parse_args(args)

    if parsed_args.verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif parsed_args.verbosity > 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig()

    for fn in parsed_args.ddd_files:
        print("read file " + fn.name)
        ddd = DDD()
        ddd.read(fn.name)
        ddd.savefig()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
