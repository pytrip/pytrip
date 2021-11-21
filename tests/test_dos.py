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

import numpy as np
import pytest

from pytrip import pytriplib
from pytrip.dos import DosCube
from pytrip.vdx import create_sphere
from pytrip.volhist import VolHist

logger = logging.getLogger(__name__)


def test_read(dos_filename):
    c = DosCube()
    c.read(dos_filename)
    assert c.cube.shape[0] == 300
    assert c.cube.shape[1] == 512
    assert c.cube.shape[2] == 512

    # test method from C extension
    dose_center = pytriplib.calculate_dose_center(np.array(c.cube))
    assert dose_center.shape[0] == 3
    assert dose_center[0] > 0.
    assert dose_center[1] > 0.
    assert dose_center[2] > 0.


def test_dvh(dos_filename):
    c = DosCube()
    c.read(dos_filename)
    v = create_sphere(c, name="sph", center=[10, 10, 10], radius=8)
    assert v is not None

    logger.info("Calculating DVH")
    result = c.calculate_dvh(v)
    assert result is not None
    dvh, min_dose, max_dose, _, area = result
    assert area > 2.0
    assert len(dvh.shape) == 2
    assert dvh.shape[1] == 2
    assert dvh.shape[0] == 1500
    assert min_dose == 0.
    assert max_dose == 0.001


def test_dvh_simple(dos_filename):
    c = DosCube()
    c.read(dos_filename)
    v = create_sphere(c, name="sph", center=[10, 10, 10], radius=8)
    assert v is not None

    logger.info("Calculating DVH simple")
    vh = VolHist(c, v)
    assert vh.x is not None
    assert vh.y is not None

    outdir = tempfile.mkdtemp()
    c.write_dicom(outdir)

    f1 = os.path.join(outdir, "foobar_1.dvh")
    vh.write(f1, header=True)
    assert os.path.exists(f1)
    assert os.path.getsize(f1) > 1

    f2 = os.path.join(outdir, "foobar_2.dvh")
    vh.write(f2, header=False)
    assert os.path.exists(f2)
    assert os.path.getsize(f2) > 1

    logger.info("Calculating DVH simple for entire cube")
    vh = VolHist(c)
    assert vh.x is not None
    assert vh.y is not None

    f3 = os.path.join(outdir, "foobar_3.dvh")
    vh.write(f3, header=True)
    assert os.path.exists(f3)
    assert os.path.getsize(f3) > 1

    f4 = os.path.join(outdir, "foobar_4.dvh")
    vh.write(f4, header=False)
    assert os.path.exists(f4)
    assert os.path.getsize(f4) > 1

    shutil.rmtree(outdir)
    # TODO: add some quantitative tests


def test_dicom_plan(dos_filename):
    c = DosCube()
    c.read(dos_filename)

    dp = c.create_dicom_plan()
    assert dp is not None

    d = c.create_dicom()
    assert d is not None


def test_write_dicom(dos_filename):
    c = DosCube()
    c.read(dos_filename)

    outdir = tempfile.mkdtemp()
    c.write_dicom(outdir)
    assert os.path.exists(os.path.join(outdir, "rtdose.dcm")) is True
    assert os.path.exists(os.path.join(outdir, "rtplan.dcm")) is True
    assert os.path.getsize(os.path.join(outdir, "rtdose.dcm")) > 1
    assert os.path.getsize(os.path.join(outdir, "rtplan.dcm")) > 1
    shutil.rmtree(outdir)


@pytest.mark.smoke
def test_write(dos_filename):
    c = DosCube()
    c.read(dos_filename)

    fd, outfile = tempfile.mkstemp()
    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file
    c.write(outfile)
    hed_file = outfile + DosCube.header_file_extension
    dos_file = outfile + DosCube.data_file_extension
    assert os.path.exists(hed_file) is True
    assert os.path.exists(dos_file) is True
    logger.info("Checking if output file " + hed_file + " is not empty")
    assert os.path.getsize(hed_file) > 1
    logger.info("Checking if output file " + dos_file + " is not empty")
    assert os.path.getsize(dos_file) > 1
    os.remove(hed_file)
    os.remove(dos_file)
