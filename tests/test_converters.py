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

import pytest

import pytrip.utils.bevlet2oer
import pytrip.utils.cubeslice
import pytrip.utils.dicom2trip
import pytrip.utils.gd2agr
import pytrip.utils.gd2dat
import pytrip.utils.rst2sobp
import pytrip.utils.trip2dicom

logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def bev_gd_filename():
    return os.path.join('tests', 'res', 'TST003', 'tst003001.bev.gd')


@pytest.fixture(scope='module')
def bevlet_gd_filename():
    return os.path.join('tests', 'res', 'TST003', 'tst003001.bevlet.gd')


def test_trip2dicom(ctx_corename):
    # create temp dir
    tmpdir = tempfile.mkdtemp()

    # convert CT cube to DICOM
    pytrip.utils.trip2dicom.main([ctx_corename, tmpdir])

    # check if destination directory is not empty
    assert len(os.listdir(tmpdir)) == 301

    shutil.rmtree(tmpdir)


def test_rst2sobp(rst_filename):
    fd, outfile = tempfile.mkstemp()

    # convert CT cube to DICOM
    pytrip.utils.rst2sobp.main([rst_filename, outfile])

    # check if destination file is not empty
    assert os.path.exists(outfile) is True
    assert os.path.getsize(outfile) > 1

    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file


def test_gd2dat(bev_gd_filename):
    fd, outfile = tempfile.mkstemp()

    # convert gd file to dat
    pytrip.utils.gd2dat.main([bev_gd_filename, outfile])

    # check if destination file is not empty
    assert os.path.exists(outfile) is True
    assert os.path.getsize(outfile) > 1

    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file


def test_gd2agr(bev_gd_filename):
    fd, outfile = tempfile.mkstemp()

    # convert gd file to agr
    pytrip.utils.gd2agr.main([bev_gd_filename, outfile])

    # check if destination file is not empty
    assert os.path.exists(outfile) is True
    assert os.path.getsize(outfile) > 1

    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file


def test_gd2oar(bevlet_gd_filename):
    fd, outfile = tempfile.mkstemp()

    # convert bev let file to oar
    pytrip.utils.bevlet2oer.main([bevlet_gd_filename, outfile])

    # check if destination file is not empty
    assert os.path.exists(outfile) is True
    assert os.path.getsize(outfile) > 1

    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file

# class TestCubeSlicer(unittest.TestCase):
#     def setUp(self):
#         self.dir_path = os.path.join("tests", "res", "TST003")
#
#         self.ctx = os.path.join(self.dir_path, "tst003000.ctx.gz")
#         logger.info("Loading ctx file " + self.ctx)
#
#         self.dos = os.path.join(self.dir_path, "tst003001.dos.gz")
#         logger.info("Loading dos file " + self.dos)
#
#         self.let = os.path.join(self.dir_path, "tst003001.dosemlet.dos.gz")
#         logger.info("Loading dos file " + self.dos)
#
#     def test_help(self):
#         try:
#             pytrip.utils.cubeslice.main(["--help"])
#         except SystemExit as e:
#             self.assertEqual(e.code, 0)
#
#     def test_version(self):
#         try:
#             pytrip.utils.cubeslice.main(["--version"])
#         except SystemExit as e:
#             self.assertEqual(e.code, 0)
#
#     def test_noarg(self):
#         try:
#             pytrip.utils.cubeslice.main([])
#         except SystemExit as e:
#             self.assertEqual(e.code, 2)
#
#     @pytest.mark.slow
#     def test_convert_all(self):
#         working_dir = tempfile.mkdtemp()  # make temp working dir for converter output files
#
#         pytrip.utils.cubeslice.main(args=['--data', self.dos, '--ct', self.ctx, '-o', working_dir])
#         output_file_list = glob.glob(os.path.join(working_dir, "*.png"))
#
#         logger.info("Checking if number of output files is sufficient")
#         self.assertEqual(len(output_file_list), 300)
#
#         for output_file in output_file_list:
#             logger.info("Checking if " + output_file + " is PNG")
#             self.assertEqual(imghdr.what(output_file), 'png')
#
#         logger.info("Removing " + working_dir)
#         shutil.rmtree(working_dir)
#
#     @pytest.mark.smoke
#     def test_convert_one(self):
#         working_dir = tempfile.mkdtemp()  # make temp working dir for converter output files
#
#         input_args = ['--data', self.dos, '--ct', self.ctx, '-f', '5', '-t', '5', '-o', working_dir]
#         ret_code = pytrip.utils.cubeslice.main(args=input_args)
#         self.assertEqual(ret_code, 0)
#
#         output_file_list = glob.glob(os.path.join(working_dir, "*.png"))
#
#         logger.info("Checking if number of output files is sufficient")
#         self.assertEqual(len(output_file_list), 1)
#
#         for output_file in output_file_list:
#             logger.info("Checking if " + output_file + " is PNG")
#             self.assertEqual(imghdr.what(output_file), 'png')
#
#         logger.info("Removing " + working_dir)
#         shutil.rmtree(working_dir)
