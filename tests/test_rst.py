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
import logging
import os
import tempfile

import pytest

import pytrip.utils.rst2sobp
from pytrip.raster import Rst

logger = logging.getLogger(__name__)


@pytest.mark.smoke
def test_read(rst_filename):
    """Check if we are able to read a simple .rst file"""
    r = Rst()
    r.read(rst_filename)
    assert r.submachines == '17'
    assert r.machines[0].points == 323
    assert r.machines[0].energy == 120.2
    assert r.machines[0].raster_points[0] == [27.0, -24.0, 2844850.0]


@pytest.mark.smoke
def test_generate(rst_filename):
    """Execute rst2sobp.py and make sure a non-empty file exists."""
    fd, outfile = tempfile.mkstemp()

    pytrip.utils.rst2sobp.main(args=[rst_filename, outfile])

    # check if destination file is not empty
    assert os.path.exists(outfile) is True
    assert os.path.getsize(outfile) > 1

    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file
