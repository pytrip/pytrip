#! /usr/bin/env python
#
#    Copyright (C) 2010-2016 PyTRiP98 Developers.
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
TODO: documentation here.
"""
import sys
import argparse
import logging

import pytrip as pt
from pytrip.utils.rst_read import RstfileRead


def main(args=sys.argv[1:]):
    # there are some cases when this script is run on systems without DISPLAY variable being set
    # in such case matplotlib backend has to be explicitly specified
    # we do it here and not in the top of the file, as inteleaving imports with code lines is discouraged
    import matplotlib
    matplotlib.use('Agg')
    from pylab import plt, ylabel, grid, xlabel, array

    parser = argparse.ArgumentParser()
    parser.add_argument("rst_file", help="location of rst file in TRiP98 format", type=str)
    parser.add_argument("output_file", help="location of PNG file to save", type=str)
    parser.add_argument("-s", "--submachine", help="Select submachine to plot.", type=int, default=1)
    parser.add_argument("-f", "--factor", help="Factor for scaling the blobs. Default is 1000.", type=int, default=1000)
    parser.add_argument("-v", "--verbosity", action='count', help="increase output verbosity", default=0)
    parser.add_argument('-V', '--version', action='version', version=pt.__version__)
    args = parser.parse_args(args)

    file = args.rst_file
    sm = args.submachine
    fac = args.factor

    a = RstfileRead(file)

    # convert data in submachine to a nice array
    b = a.submachine[sm]
    title = "Submachine: {:d} / {:d} - Energy: {:.3f} MeV/u".format(sm, len(a.submachine), b.energy)
    print(title)
    cc = array(b.particles)

    cc = cc / cc.max() * fac

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(b.xpos, b.ypos, c=cc, s=cc, alpha=0.75)
    ylabel("mm")
    xlabel("mm")
    grid(True)
    plt.title(title)
    plt.savefig(args.output_file)
    plt.close()

if __name__ == '__main__':
    logging.basicConfig()
    sys.exit(main(sys.argv[1:]))
