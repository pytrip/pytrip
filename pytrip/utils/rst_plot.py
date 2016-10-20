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
import logging
from optparse import OptionParser

from pytrip.utils.rst_read import RstfileRead


def main(args=sys.argv[1:]):
    # there are some cases when this script is run on systems without DISPLAY variable being set
    # in such case matplotlib backend has to be explicitly specified
    # we do it here and not in the top of the file, as inteleaving imports with code lines is discouraged
    import matplotlib
    matplotlib.use('Agg')
    from pylab import plt, ylabel, grid, xlabel, array

    parser = OptionParser()
    parser.add_option("-s", "--submachine", dest="subm", help="Select submachine to plot.", metavar="int")
    parser.add_option(
        "-f", "--factor", dest="fac", help="Factor for scaling the blobs. "
        "Default is 1000.", metavar="int")
    (options, args) = parser.parse_args(args)

    file = args[0]

    sm = 1  # default
    fac = 1000
    if options.subm is not None:
        sm = int(options.subm)
    if options.fac is not None:
        fac = int(options.fac)

    a = RstfileRead(file)

    # convert data in submachine to a nice array
    b = a.submachine[sm]
    print("Submachine: ", sm, " - Energy:", b.energy, "MeV/u")
    cc = array(b.particles)

    cc = cc / cc.max() * fac

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(b.xpos, b.ypos, c=cc, s=cc, alpha=0.75)
    ylabel("mm")
    xlabel("mm")

    grid(True)

    plt.show()


if __name__ == '__main__':
    logging.basicConfig()
    sys.exit(main(sys.argv[1:]))
