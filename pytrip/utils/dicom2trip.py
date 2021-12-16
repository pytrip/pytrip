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
Script for converting a DICOM file to TRiP98 / Voxelplan style files.
"""
import sys
import logging
import argparse

import pytrip as pt

logger = logging.getLogger(__name__)


def main(args=None):
    """ Main function for dicom2trip.py
    """
    if args is None:
        args = sys.argv[1:]

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dicom_dir", help="location of directory with DICOM files", type=str)
    parser.add_argument("ctx_basename", help="basename of output file in TRiP98 format", type=str)
    parser.add_argument("-v", "--verbosity", action='count', help="increase output verbosity", default=0)
    parser.add_argument('-V', '--version', action='version', version=pt.__version__)
    parsed_args = parser.parse_args(args)

    if parsed_args.verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif parsed_args.verbosity > 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig()

    basename = parsed_args.ctx_basename

    # import DICOM
    dcm = pt.dicomhelper.read_dicom_dir(parsed_args.dicom_dir)

    if 'images' in dcm:
        c = pt.CtxCube()
        c.read_dicom(dcm)
        logger.info("Write CtxCube ... '{:s}'".format(basename + pt.CtxCube.data_file_extension))
        c.write(basename)
    else:
        logger.warning("No CT data found in {:s}".format(parsed_args.dicom_dir))
        c = None

    if 'rtss' in dcm:
        logger.info("Write VdxCube ... '{:s}'".format(basename + ".vdx"))
        vdx_cube = pt.VdxCube(cube=c)
        vdx_cube.read_dicom(dcm)
        vdx_cube.write_trip(basename + ".vdx")
    else:
        logger.warning("No RTSTRUCT data found in {:s}".format(parsed_args.dicom_dir))

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
