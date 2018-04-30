#!/usr/bin/env python
#
#    Copyright (C) 2010-2018 PyTRiP98 Developers.
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

import pytrip as pt
from pytrip import spc


def main(args=sys.argv[1:]):

    # matplotlib on python3.2 (which is already deprecated) won't handle this code
    # it is not worthy to hack it for this particular piece of the code
    if sys.version_info[0] == 3 and sys.version_info[1] == 2:
        logging.error("Python 3.2 is not supported, please use other version")
        return 1

    # there are some cases (i.e. Travis CI) when this script is run on systems without DISPLAY variable being set
    # in such case matplotlib backend has to be explicitly specified
    # we do it here and not in the top of the file, as interleaving imports with code lines is discouraged
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.backends.backend_pdf import PdfPages

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("spc_file", help="location of input SPC file", type=argparse.FileType('r'))
    parser.add_argument("pdf_file", help="location of output PDF file", type=argparse.FileType('w'))
    parser.add_argument('-d', '--depth', help="Plot fluence spectra at depths",
                        type=float, nargs='+')
    parser.add_argument('-l', '--logscale', help="Enable fluence plotting on logarithmic scale", action='store_true')
    parser.add_argument('-c', '--colormap', help='image color map, see http://matplotlib.org/users/colormaps.html '
                                                 'for list of possible options (default: gnuplot2)',
                        default='gnuplot2', type=str)
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
    for db in sorted(spc_object.data, key=lambda x: x.depth):
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
    for db in sorted(spc_object.data, key=lambda x: x.depth):
        for sb in sorted(db.species, key=lambda y: y.z):
            data.depth[n:n + int(sb.ne)] = db.depth
            data.z[n:n + int(sb.ne)] = sb.z
            data.energy[n:n + int(sb.ne)] = sb.ebindata[:-1]  # left edges of bins
            data.fluence[n:n + int(sb.ne)] = sb.histdata
            n += int(sb.ne)

    logging.debug("Temporary numpy array filled with data, size {}".format(data.size))

    # find particle species, depth steps and energy bins
    # TODO think of switching in the future to pandas dataframe with mutliple indexes
    z_uniq = np.unique(data.z)
    logging.info("Particle species Z: {}".format(z_uniq))

    depth_uniq = np.unique(data.depth)
    if np.unique(np.diff(depth_uniq)).size == 1:
        logging.info("{} of depth steps from {} to {}".format(len(depth_uniq), depth_uniq.min(), depth_uniq.max()))
    else:
        logging.info("Depth steps {}".format(depth_uniq))

    energy_uniq = np.unique(data.energy)
    if np.unique(np.diff(energy_uniq)).size == 1:
        logging.info("{} of depth energy bins from {} to {} (left edges)".format(
            len(energy_uniq), energy_uniq.min(), energy_uniq.max()))
    else:
        logging.info("Energy steps {}".format(energy_uniq))

    # save multiple page PDF with plots
    with PdfPages(parsed_args.pdf_file.name) as pdf:
        logging.debug("Saving PDF file {}".format(parsed_args.pdf_file.name))
        plt.rc('text', usetex=False)

        # first a summary plot of fluence vs depth, several series corresponding to Z species
        fig, ax = plt.subplots()
        ax.set_title("")
        ax.set_xlabel("Depth [cm]")
        ax.set_ylabel("Fluence [a.u.]")
        if parsed_args.logscale:
            ax.set_yscale('log')
        for z in z_uniq:
            z_data = data[data.z == z]  # this may cover depth only partially
            if z_data.fluence.any():
                depth_steps = np.unique(data.depth[data.z == z])
                energy_steps = np.unique(data.energy[data.z == z])
                zlist = z_data.fluence.reshape(depth_steps.size, energy_steps.size).T
                total_fluence = zlist.sum(axis=0)
                ax.plot(depth_steps, total_fluence, label="Z = {}".format(z))
        plt.legend(loc=0)
        pdf.savefig(fig)
        plt.close()
        logging.debug("Summary fluence vs depth plot saved")

        if parsed_args.depth:
            # then a spectrum plot at depths selected by user
            for depth in parsed_args.depth:
                fig, ax = plt.subplots()
                ax.set_title("Energy spectrum at depth {} [cm]".format(depth))
                ax.set_xlabel("Energy [MeV]")
                ax.set_ylabel("Fluence [a.u.]")
                if parsed_args.logscale:
                    ax.set_yscale('log')
                for z in z_uniq:
                    z_data = data[data.z == z]  # this may cover depth only partially
                    depth_step = data.depth[np.abs(data.depth - depth).argmin()]
                    if z_data.fluence.any():
                        m1 = (data.z == z)
                        m2 = (data.depth == depth_step)
                        m3 = m1 & m2
                        energy_steps = data.energy[m3]
                        fluence = data.fluence[m3]
                        if np.any(m3):
                            ax.plot(energy_steps, fluence, label="Z = {:d}, depth = {:3.3f} [cm]".format(z, depth_step))
                plt.legend(loc=0)
                pdf.savefig(fig)
                plt.close()
                logging.debug("Spectrum for depth {} saved".format(depth_step))

        # then couple of pages, each with heatmap plot of spectrum for given particle specie
        for z in z_uniq:
            z_data = data[data.z == z]
            if z_data.fluence.any():
                depth_steps = np.unique(data.depth[data.z == z])
                energy_steps = np.unique(data.energy[data.z == z])
                zlist = z_data.fluence.reshape(depth_steps.size, energy_steps.size).T
                if parsed_args.logscale:
                    norm = colors.LogNorm(vmin=zlist[zlist > 0.0].min(), vmax=zlist.max())
                else:
                    norm = colors.Normalize(vmin=0.0, vmax=zlist.max())
                fig, ax = plt.subplots()
                ax.set_title("Spectrum for Z = {}".format(z))
                ax.set_xlabel("Depth [cm]")
                ax.set_ylabel("Energy [MeV]")
                max_energy_nonzero_fluence = energy_steps[zlist.mean(axis=1) > 0].max()
                ax.set_ylim(0, 1.2 * max_energy_nonzero_fluence)

                # in case depth or energy steps form arithmetic progress, then pcolorfast expects
                #   third argument (Z) to be of shape len(X) times len(Y)
                # otherwise it expects
                #   third argument (Z) to be of shape len(X)-1 times len(Y)-1
                # here we check if latter is the case and add one element to X and Y sequences
                depth_steps_widths = np.diff(depth_steps)
                energy_steps_widths = np.diff(energy_steps)
                if np.unique(depth_steps_widths).size > 1 or np.unique(energy_steps_widths).size > 1:
                    depth_steps = np.append(depth_steps, depth_steps[-1] + depth_steps_widths.min())
                    energy_steps = np.append(energy_steps, energy_steps[-1] + energy_steps_widths.min())

                im = ax.pcolorfast(depth_steps, energy_steps, zlist, norm=norm, cmap=parsed_args.colormap)
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
