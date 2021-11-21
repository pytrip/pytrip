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
import gzip
import hashlib
import logging
import os
import tempfile

import pytest

from pytrip.ctx import CtxCube
from pytrip.error import FileNotFound
from pytrip.util import TRiP98FileLocator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_read(ctx_corename):
    c = CtxCube()
    c.read(ctx_corename)


@pytest.mark.slow
def test_read_and_write_cube(ctx_allnames):
    logger.info("Testing cube from path " + ctx_allnames)

    # read original cube and calculate hashsum
    c = CtxCube()
    c.read(ctx_allnames)

    # get path to the cube data file, extracting it from a partial path
    data_file_path = TRiP98FileLocator(ctx_allnames, CtxCube).datafile

    # get the hashsum
    if data_file_path.endswith(".gz"):
        f = gzip.open(data_file_path)
    else:
        f = open(data_file_path, 'rb')
    original_md5 = hashlib.md5(f.read()).hexdigest()
    f.close()

    # calculate temporary filename
    fd, outfile = tempfile.mkstemp()
    os.close(fd)
    os.remove(outfile)  # we need only random name, not a descriptor
    logger.debug("Generated random file name " + outfile)

    # save cube and calculate hashsum
    saved_header_path, saved_cubedata_path = c.write(outfile)  # this will write outfile+".ctx"  and outfile+".hed"

    # check if generated files exists
    assert os.path.exists(saved_header_path)
    assert os.path.exists(saved_cubedata_path)

    # get checksum
    f = open(saved_cubedata_path, 'rb')
    generated_md5 = hashlib.md5(f.read()).hexdigest()
    f.close()
    logger.debug("Removing " + saved_cubedata_path)
    os.remove(saved_cubedata_path)
    logger.debug("Removing " + saved_header_path)
    os.remove(saved_header_path)
    # compare checksums
    assert original_md5 == generated_md5


def test_problems_when_reading(ctx_corename):
    # check malformed filename
    with pytest.raises(FileNotFound) as e:
        logger.info("Catching {:s}".format(str(e)))
        test_read_and_write_cube(ctx_corename[2:-2])

    # check exception if filename is without dot
    with pytest.raises(FileNotFound) as e:
        logger.info("Catching {:s}".format(str(e)))
        test_read_and_write_cube(ctx_corename + "hed")

    # check opening wrong filetype (file self.cube000 + ".vdx" exists !)
    with pytest.raises(FileNotFound) as e:
        logger.info("Catching {:s}".format(str(e)))
        test_read_and_write_cube(ctx_corename + ".vdx")


@pytest.mark.smoke
def test_addition(ctx_corename):
    # read cube
    c = CtxCube()
    c.read(ctx_corename)
    d = c + 5
    assert c.cube[10][20][30] + 5 == d.cube[10][20][30]
