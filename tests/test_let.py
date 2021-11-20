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
import tempfile

from pytrip.let import LETCube
from pytrip.vdx import create_sphere
from tests.conftest import exists_and_nonempty

logger = logging.getLogger(__name__)


def test_read(let_filename):
    let = LETCube()
    let.read(let_filename)

    v = create_sphere(let, name="sph", center=[10., 10., 10.], radius=8.)
    assert v is not None

    logger.info("Calculating DVH")
    result = let.calculate_lvh(v)
    assert result is not None
    lvh, min_l, max_l, _, area = result
    assert area > 2.
    assert len(lvh.shape) == 1
    assert lvh.shape[0] == 3000
    assert min_l == 0.
    assert max_l == 1.

    assert let.get_max() > 30.

    fd, outfile = tempfile.mkstemp()
    os.close(fd)  # Windows needs it
    os.remove(outfile)  # we need only temp filename, not the file
    let.write(outfile)
    hed_file = outfile + LETCube.header_file_extension
    dos_file = outfile + LETCube.data_file_extension
    assert exists_and_nonempty(hed_file) is True
    assert exists_and_nonempty(dos_file) is True
    os.remove(hed_file)
    os.remove(dos_file)
