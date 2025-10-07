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
import glob
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
from tests.conftest import exists_and_nonempty, image_type

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

    assert exists_and_nonempty(outfile)

    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file


def test_gd2dat(bev_gd_filename):
    fd, outfile = tempfile.mkstemp()

    # convert gd file to dat
    pytrip.utils.gd2dat.main([bev_gd_filename, outfile])

    assert exists_and_nonempty(outfile)

    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file


def test_gd2agr(bev_gd_filename):
    fd, outfile = tempfile.mkstemp()

    # convert gd file to agr
    pytrip.utils.gd2agr.main([bev_gd_filename, outfile])

    assert exists_and_nonempty(outfile)

    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file


def test_gd2oar(bevlet_gd_filename):
    fd, outfile = tempfile.mkstemp()

    # convert bev let file to oar
    pytrip.utils.bevlet2oer.main([bevlet_gd_filename, outfile])

    assert exists_and_nonempty(outfile)

    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file


@pytest.mark.parametrize("option_name", ["version", "help"])
def test_call_cmd_option(option_name):
    with pytest.raises(SystemExit) as e:
        logger.info("Catching {:s}".format(str(e)))
        pytrip.utils.cubeslice.main(['--' + option_name])
        assert e.code == 0


@pytest.mark.skip("need to properly handle exception when obligatory arguments are missing")
def test_call_no_options():
    with pytest.raises(SystemExit) as e:
        logger.info("Catching {:s}".format(str(e)))
        pytrip.utils.cubeslice.main([])
        assert e.code == 2


@pytest.mark.smoke
def test_convert_one(ctx_corename, dos_filename):
    working_dir = tempfile.mkdtemp()  # make temp working dir for converter output files

    input_args = ['--data', dos_filename, '--ct', ctx_corename, '-f', '5', '-t', '5', '-o', working_dir]
    ret_code = pytrip.utils.cubeslice.main(args=input_args)
    assert ret_code == 0

    output_file_list = glob.glob(os.path.join(working_dir, "*.png"))

    logger.info("Checking if number of output files is sufficient")
    assert len(output_file_list) == 1
    output_filename = output_file_list[0]
    assert os.path.basename(output_filename) == "tst003001_005.png"

    logger.info("Checking if " + output_filename + " is PNG")
    assert image_type(output_filename) == 'png'

    logger.info("Removing " + working_dir)
    shutil.rmtree(working_dir)


@pytest.mark.slow
def test_convert_all(ctx_corename, dos_filename):
    working_dir = tempfile.mkdtemp()  # make temp working dir for converter output files

    pytrip.utils.cubeslice.main(args=['--data', dos_filename, '--ct', ctx_corename, '-o', working_dir])
    output_file_list = glob.glob(os.path.join(working_dir, "*.png"))

    logger.info("Checking if number of output files is sufficient")
    assert len(output_file_list) == 300

    for output_file in output_file_list:
        logger.info("Checking if " + output_file + " is PNG")
        assert image_type(output_file) == 'png'

    logger.info("Removing " + working_dir)
    shutil.rmtree(working_dir)
