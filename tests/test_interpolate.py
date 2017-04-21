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
import unittest
import logging
import numpy as np

import tests.test_base

logger = logging.getLogger(__name__)

from pytrip.res.interpolate import RegularInterpolator

class TestInterpolate(unittest.TestCase):

    def test_point(self):
        interp_1d = RegularInterpolator(x=[1], y=[1])
        self.assertEqual(interp_1d(1),1)
        self.assertEqual(interp_1d([1,1,1]),[1,1,1])


if __name__ == '__main__':
    unittest.main()
