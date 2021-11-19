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
"""
import logging
import os

import pytest

import pytrip.tripexecuter as pte

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope='module')
def exec_filename():
    return os.path.join('tests', 'res', 'TST003', 'EXEC', 'TST003101.exec')


def test_exec_parse(exec_filename):
    """TODO"""
    logger.info("Test parsing '{:s}'".format(exec_filename))

    plan = pte.Plan()
    plan.read_exec(exec_filename)
    assert len(plan.fields) == 3
