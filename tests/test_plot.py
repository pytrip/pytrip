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
TODO: documentation here.
"""
import logging
import os
import shutil
import tempfile

import pytrip.utils.dvhplot
import pytrip.utils.rst_plot
from tests.conftest import image_type

logger = logging.getLogger(__name__)


def test_generate(rst_filename):
    fd, outfile = tempfile.mkstemp(suffix='.png')

    # convert CT cube to DICOM
    pytrip.utils.rst_plot.main([rst_filename, outfile])

    # check if destination file is not empty
    assert os.path.exists(outfile) is True
    assert os.path.getsize(outfile) > 1
    assert image_type(outfile) == 'png'

    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file


def test_relative_dos_plot(dos_filename, vdx_filename):
    working_dir = tempfile.mkdtemp()  # make temp working dir for output file
    output_file = os.path.join(working_dir, "foo.png")

    pytrip.utils.dvhplot.main(args=[dos_filename, vdx_filename, 'target', '-l', '-v', '-o', output_file])

    logger.info("Checking if " + output_file + " is PNG")
    assert image_type(output_file) == 'png'

    logger.info("Removing " + working_dir)
    shutil.rmtree(working_dir)


def test_absolute_dos_plot(dos_filename, vdx_filename):
    working_dir = tempfile.mkdtemp()  # make temp working dir for output file
    output_file = os.path.join(working_dir, "foo.png")

    pytrip.utils.dvhplot.main(args=[dos_filename, vdx_filename, 'target', '-l', '-v', '-d 2.0', '-o', output_file])

    logger.info("Checking if " + output_file + " is PNG")
    assert image_type(output_file) == 'png'

    logger.info("Removing " + working_dir)
    shutil.rmtree(working_dir)


def test_let_plot(let_filename, vdx_filename):
    working_dir = tempfile.mkdtemp()  # make temp working dir for output file
    output_file = os.path.join(working_dir, "foo.png")

    pytrip.utils.dvhplot.main(args=[let_filename, vdx_filename, 'target', '-l', '-v', '-o', output_file])

    logger.info("Checking if " + output_file + " is PNG")
    assert image_type(output_file) == 'png'

    logger.info("Removing " + working_dir)
    shutil.rmtree(working_dir)
