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
Script for inspecting SPC files and producing PDF with summary plots
"""
import sys
import logging
import argparse

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages

import pytrip as pt
from pytrip import spc


def main(args=sys.argv[1:]):

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("spc_file", help="location of input SPC file", type=argparse.FileType('r'))
    parser.add_argument("pdf_file", help="location of output PDF file", type=argparse.FileType('w'))
    parser.add_argument("-v", "--verbosity", action='count', help="increase output verbosity", default=0)
    parser.add_argument('-V', '--version', action='version', version=pt.__version__)
    parsed_args = parser.parse_args(args)

    if parsed_args.verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif parsed_args.verbosity > 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig()

    # read SPC file
    spc_object = spc.SPC(parsed_args.spc_file.name)
    spc_object.read_spc()

    # get total number of entries in SPC object, needed to allocate memory for numpy array
    n = 0
    for db in spc_object.data:
        for sb in db.species:
            n += int(sb.ne)
    logging.info("Total number of entries in SPC file {}".format(n))

    # allocate empty numpy array to store SPC data
    data = np.recarray(
        shape=(int(n), ),
        dtype=[('depth', np.float64), ('z', np.int8), ('energy', np.float64), ('fluence', np.float64)])
    logging.debug("Temporary numpy array to store SPC data has shape {}".format(data.shape))

    # fill numpy array with spectral data, assuming that no references are used to store bin centers
    n = 0
    for db in spc_object.data:
        for sb in db.species:
            data.depth[n:n + int(sb.ne)] = db.depth
            data.z[n:n + int(sb.ne)] = sb.z
            data.energy[n:n + int(sb.ne)] = 0.5 * (sb.ebindata[:-1] + sb.ebindata[1:])  # bin centers
            data.fluence[n:n + int(sb.ne)] = sb.histdata
            n += int(sb.ne)

    logging.debug("Temporary numpy array filled with data, size {}".format(data.size))

    # find particle species, depth steps and energy bin centers
    z_uniq = np.unique(data.z)
    logging.info("Particle species Z: {}".format(z_uniq))

    depth_uniq = np.unique(data.depth)
    logging.info("{} of depth steps from {} to {}".format(len(depth_uniq), depth_uniq.min(), depth_uniq.max()))

    energy_uniq = np.unique(data.energy)
    logging.info("{} of depth energy bins from {} to {}".format(len(energy_uniq), energy_uniq.min(), energy_uniq.max()))

    # save multiple page PDF with plots
    with PdfPages(parsed_args.pdf_file.name) as pdf:
        logging.debug("Saving PDF file {}".format(parsed_args.pdf_file.name))
        plt.rc('text', usetex=False)

        # first a summary plot of fluence vs depth, log scale on fluence (Y), several series corresponding to Z species
        fig, ax = plt.subplots()
        ax.set_title("")
        ax.set_xlabel("Depth [cm]")
        ax.set_ylabel("Fluence [a.u.]")
        ax.set_yscale('log')
        for z in z_uniq:
            z_data = data[data.z == z]
            zlist = z_data.fluence.reshape(depth_uniq.size, energy_uniq.size).T
            if zlist.any():
                total_fluence = zlist.sum(axis=0)
                ax.plot(depth_uniq, total_fluence, label="Z = {}".format(z))
        plt.legend(loc=0)
        pdf.savefig(fig)
        plt.close()
        logging.debug("Summary fluence vs depth plot saved")

        # then couple of pages, each with heatmap plot of spectrum for given particle specie
        for z in z_uniq:
            z_data = data[data.z == z]
            zlist = z_data.fluence.reshape(depth_uniq.size, energy_uniq.size).T
            if zlist.any():
                norm = colors.LogNorm(vmin=zlist[zlist > 0.0].min(), vmax=zlist.max())
                fig, ax = plt.subplots()
                ax.set_title("Spectrum for Z = {}".format(z))
                ax.set_xlabel("Depth [cm]")
                ax.set_ylabel("Energy [MeV]")
                max_energy_nonzero_fluence = energy_uniq[zlist.mean(axis=1) > 0].max()
                ax.set_ylim(0, 1.2 * max_energy_nonzero_fluence)
                im = ax.pcolorfast(depth_uniq, energy_uniq, zlist, norm=norm)
                cbar = plt.colorbar(im)
                cbar.set_label("Fluence [a.u.]", rotation=270, verticalalignment='bottom')
                pdf.savefig(fig)
                plt.close()
                logging.debug("Fluence map for Z = {} saved".format(z))
            else:
                logging.warning("Skipped generation of fluence map for Z = {}, no data !".format(z))

    logging.info("Done")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
