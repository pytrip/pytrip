#!/usr/bin/env python
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
Script for converting Voxelplan / TRiP98 Ctx and Vdx data to a DICOM file.
"""
import os
import sys
import logging
import argparse

import pytrip as pt

logger = logging.getLogger(__name__)


def main(args=sys.argv[1:]):
    """ Main function for trip2dicom.py
    """
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("ctx_data", help="location of CT file (header or data) in TRiP98 format", type=str)
    parser.add_argument("outputdir", help="write resulting DICOM files to this directory", type=str)
    parser.add_argument("-v", "--verbosity", action='count', help="increase output verbosity", default=0)
    parser.add_argument('-V', '--version', action='version', version=pt.__version__)
    parsed_args = parser.parse_args(args)

    if parsed_args.verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif parsed_args.verbosity > 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig()

    output_folder = parsed_args.outputdir

    _, data_file_name = pt.CtxCube.parse_path(parsed_args.ctx_data)
    data_file_path = pt.CtxCube.discover_file(data_file_name)

    if not os.path.exists(data_file_path):
        logger.error("CTX file missing")
        return 1

    logger.info("Convert CT images...")
    c = pt.CtxCube()
    c.read(data_file_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    c.write_dicom(output_folder)

    ctx_basename = os.path.splitext(data_file_path)[0]
    ctx_path = ctx_basename + ".vdx"
    if os.path.exists(ctx_path):
        logger.info("Convert VDX structures...")
        v = pt.VdxCube(cube=c)
        v.read(ctx_path)
        v.write_dicom(output_folder)
    else:
        logger.info("No VDX data found for conversion.")

    logger.info("Done")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
