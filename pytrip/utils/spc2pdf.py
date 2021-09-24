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
import argparse
import datetime
import getpass
import logging
import math
import sys

import numpy as np

import pytrip as pt
from pytrip import spc


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # matplotlib on python3.2 (which is already deprecated) won't handle this code
    # it is not worthy to hack it for this particular piece of the code
    if sys.version_info[0] == 3 and sys.version_info[1] == 2:
        logging.error("Python 3.2 is not supported, please use other version")
        return 1

    # there are some cases (i.e. continuous automation) when this script is run without DISPLAY variable being set
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
    parser.add_argument('-d', '--depth', help="Plot spectra at depths", type=float, nargs='+')
    parser.add_argument('-l',
                        '--logscale',
                        help="Enable plotting particle number on logarithmic scale",
                        action='store_true')
    parser.add_argument('-s', '--style', help="Line style", choices=['points', 'lines'], default='lines')
    parser.add_argument('-c',
                        '--colormap',
                        help='image color map, see '
                        'https://matplotlib.org/stable/tutorials/colors/colormaps.html '
                        'for list of possible options (default: gnuplot2)',
                        default='gnuplot2',
                        type=str)
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
    data = np.recarray(shape=(int(n), ),
                       dtype=[('depth', np.float64), ('z', np.int8), ('a', np.int8), ('energy_left_edge', np.float64),
                              ('energy_bin_width', np.float64), ('tot_part_number', np.float64)])
    logging.debug("Temporary numpy array to store SPC data has shape {}".format(data.shape))

    # fill numpy array with spectral data, assuming that no references are used to store bin centers
    n = 0
    for db in sorted(spc_object.data, key=lambda x: x.depth):
        for sb in sorted(db.species, key=lambda y: y.z):
            data.depth[n:n + int(sb.ne)] = db.depth
            data.z[n:n + int(sb.ne)] = sb.z
            data.a[n:n + int(sb.ne)] = sb.a
            data.energy_left_edge[n:n + int(sb.ne)] = sb.ebindata[:-1]  # left edges of bins
            data.energy_bin_width[n:n + int(sb.ne)] = np.diff(sb.ebindata)  # bin widths
            data.tot_part_number[n:n + int(sb.ne)] = sb.histdata
            n += int(sb.ne)

    logging.debug("Temporary numpy array filled with data, size {}".format(data.size))

    # find particle species, depth steps and energy bins
    az_uniq = np.unique(data[['a', 'z']])
    logging.info("Particle species A, Z: {}".format(az_uniq))

    depth_uniq = np.unique(data.depth)
    if np.unique(np.diff(depth_uniq)).size == 1:
        logging.info("{} of depth steps from {} to {}".format(len(depth_uniq), depth_uniq.min(), depth_uniq.max()))
    else:
        logging.info("Depth steps {}".format(depth_uniq))

    energy_uniq = np.unique(data.energy_left_edge)
    if np.unique(np.diff(energy_uniq)).size == 1:
        logging.info("{} of depth energy bins from {} to {} (left edges)".format(len(energy_uniq), energy_uniq.min(),
                                                                                 energy_uniq.max()))
    else:
        logging.info("Energy steps {}".format(energy_uniq))

    # save multiple page PDF with plots
    with PdfPages(parsed_args.pdf_file.name) as pdf:
        logging.debug("Saving PDF file {}".format(parsed_args.pdf_file.name))
        plt.rc('text', usetex=False)

        # choose line style
        linestyle = '-'
        marker = ''
        if parsed_args.style == 'lines':
            linestyle = '-'
            marker = ''
        elif parsed_args.style == 'points':
            linestyle = ''
            marker = '.'

        # first a summary plot of particle number vs depth, several series corresponding to Z species
        fig, ax = plt.subplots()
        ax.set_title("Peak position {:3.3f} [cm], beam energy {} MeV/amu".format(spc_object.peakpos, spc_object.energy))
        ax.set_xlabel("Depth [cm]")
        ax.set_ylabel("Particle number / primary [(none)]")
        if parsed_args.logscale:
            ax.set_yscale('log')
        for az in az_uniq:
            a, z = az
            az_data = data[(data.z == z) & (data.a == a)]  # this may cover depth only partially
            if az_data.tot_part_number.any():
                depth_steps = az_data.depth[az_data.energy_left_edge == az_data.energy_left_edge[0]]
                energy_bin_widths = az_data.energy_bin_width[az_data.depth == az_data.depth[0]]
                zlist = az_data.tot_part_number.reshape(depth_steps.size, energy_bin_widths.size).T
                part_number_at_depth = (zlist.T * energy_bin_widths).sum(axis=1)
                ax.plot(depth_steps,
                        part_number_at_depth,
                        marker=marker,
                        linestyle=linestyle,
                        label="A = {}, Z = {}".format(a, z))
        plt.legend(loc=0)
        pdf.savefig(fig)
        plt.close()
        logging.debug("Summary particle number vs depth plot saved")

        if parsed_args.depth:
            # then a spectrum plot at depths selected by user
            depths_to_plot = parsed_args.depth
            if any(math.isnan(d) for d in parsed_args.depth):
                depths_to_plot = depth_uniq
                logging.info("Plotting spectrum at all {} depths requested".format(len(depths_to_plot)))
            for depth_requested in depths_to_plot:
                depth_step = data.depth[np.abs(data.depth - depth_requested).argmin()]
                logging.info("Depth step {}, depth requested {}".format(depth_step, depth_requested))
                fig, ax = plt.subplots()
                ax.set_title("Spectrum @ {:3.3f} cm".format(depth_step))
                ax.set_xlabel("Energy [MeV/amu]")
                ax.set_ylabel("Particle number / primary [1/MeV/amu]")
                if parsed_args.logscale:
                    ax.set_yscale('log')
                for az in az_uniq:
                    a, z = az
                    az_data = data[(data.z == z) & (data.a == a)]
                    if az_data.tot_part_number.any():
                        mask1 = (az_data.depth == depth_step)
                        energy_bin_centers = az_data.energy_left_edge[mask1] + 0.5 * az_data.energy_bin_width[mask1]
                        tot_part_number = az_data.tot_part_number[mask1]
                        if np.any(mask1):
                            ax.plot(energy_bin_centers,
                                    tot_part_number,
                                    linestyle=linestyle,
                                    marker=marker,
                                    label="Z = {:d} A = {:d}".format(z, a))
                plt.legend(loc=0)
                pdf.savefig(fig)
                plt.close()
                logging.debug("Spectrum for depth {} saved".format(depth_step))

        # then couple of pages, each with heatmap plot of spectrum for given particle specie
        for az in az_uniq:
            a, z = az
            az_data = data[(data.z == z) & (data.a == a)]
            if az_data.tot_part_number.any():
                depth_steps = az_data.depth[az_data.energy_left_edge == az_data.energy_left_edge[0]]
                energy_left_edges = az_data.energy_left_edge[az_data.depth == az_data.depth[0]]
                energy_bin_widths = az_data.energy_bin_width[az_data.depth == az_data.depth[0]]

                zlist = az_data.tot_part_number.reshape(depth_steps.size, energy_left_edges.size).T
                if parsed_args.logscale:
                    norm = colors.LogNorm(vmin=zlist[zlist > 0.0].min(), vmax=zlist.max())
                else:
                    norm = colors.Normalize(vmin=0.0, vmax=zlist.max())
                fig, ax = plt.subplots()
                ax.set_title("Spectrum for Z = {} A = {}".format(z, a))
                ax.set_xlabel("Depth [cm]")
                ax.set_ylabel("Energy [MeV/amu]")
                max_energy_nonzero_part_no = energy_left_edges[zlist.mean(axis=1) > 0].max()
                ax.set_ylim(0, 1.2 * max_energy_nonzero_part_no)

                # in case depth or energy steps form arithmetic progress, then pcolorfast expects
                #   third argument (Z) to be of shape len(X) times len(Y)
                # otherwise it expects
                #   third argument (Z) to be of shape len(X)-1 times len(Y)-1
                # here we check if latter is the case and add one element to X and Y sequences
                energy_steps = energy_left_edges
                depth_steps_widths = np.diff(depth_steps)
                if np.unique(depth_steps_widths).size > 1 or np.unique(energy_bin_widths).size > 1:
                    depth_steps = np.append(depth_steps, depth_steps[-1] + depth_steps_widths.min())
                    energy_steps = np.append(energy_left_edges, energy_left_edges[-1] + energy_bin_widths[-1])

                im = ax.pcolorfast(depth_steps, energy_steps, zlist, norm=norm, cmap=parsed_args.colormap)
                cbar = plt.colorbar(im)
                cbar.set_label("Particle number / primary [1/MeV/amu]", rotation=270, verticalalignment='bottom')
                pdf.savefig(fig)
                plt.close()
                logging.debug("Particle number / primary map for Z = {} saved".format(z))
            else:
                logging.warning("Skipped generation of particle number map for Z = {}, no data !".format(z))

        # File metadata
        d = pdf.infodict()
        d['Title'] = 'SPC summary for {}'.format(parsed_args.spc_file.name)
        d['Author'] = 'User {}'.format(getpass.getuser())
        d['Subject'] = '{}'.format(" ".join(sys.argv))
        d['Keywords'] = 'pytrip, spc'
        d['Creator'] = 'pytrip98 {}'.format(pt.__version__)
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()

    logging.info("Done")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
