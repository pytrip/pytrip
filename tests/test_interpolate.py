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
import unittest
import logging
import numpy as np
from numpy.testing import assert_allclose

from pytrip.res.interpolate import RegularInterpolator

logger = logging.getLogger(__name__)


class TestInterpolate(unittest.TestCase):
    def test_point(self):
        """Single data point
        """
        for kind in ('linear', 'spline'):
            interp_1d = RegularInterpolator(x=(1.,), y=(2.,), kind=kind)
            self.assertEqual(interp_1d(1), 2)
            self.assertEqual(RegularInterpolator.eval(x=1, xp=(1.,), yp=(2.,), kind=kind), 2)

            self.assertEqual(interp_1d(5), 2)  # value outside domain
            self.assertEqual(RegularInterpolator.eval(x=5, xp=(1.,), yp=(2.,), kind=kind), 2)

            assert_allclose(interp_1d(x=(1, 1, 1)), (2, 2, 2))
            assert_allclose(interp_1d(x=(0, 1, 2)), (2, 2, 2))  # some values outside domain
            assert_allclose(interp_1d(np.array((1, 1, 1))), (2, 2, 2))  # numpy array
            assert_allclose(interp_1d(()), ())  # empty table

            interp_2d = RegularInterpolator(x=(1.,), y=(2.,), z=(3.,), kind=kind)
            self.assertEqual(interp_2d(1, 2), 3)
            self.assertEqual(interp_2d(5, 0), 3)  # value outside domain
            assert_allclose(interp_2d((1, 1, 1), (2, 2, 2)), (3, 3, 3))
            test_y = RegularInterpolator.eval(x=(1, 1, 1), y=(2, 2, 2), xp=(1.,), yp=(2.,), zp=(3.,), kind=kind)
            assert_allclose(test_y, (3, 3, 3))
            assert_allclose(interp_2d((0, 1, 2), (0, 1, 2)), (3, 3, 3))  # some values outside domain
            assert_allclose(interp_2d(np.array((1, 1, 1)), np.array((2, 2, 2))), (3, 3, 3))  # numpy array
            assert_allclose(interp_2d((), ()), ())  # empty table

    def test_1d_2el(self):
        """
        Should fallback to linear interpolation
        """
        for kind in ('linear', 'spline'):
            interp_1d = RegularInterpolator(x=(1., 3.), y=(-2.0, 2.0), kind=kind)
            self.assertEqual(interp_1d(1), -2)
            self.assertEqual(interp_1d(1.25), -1.5)
            self.assertEqual(interp_1d(1.5), -1)
            self.assertEqual(interp_1d(2), 0)
            self.assertEqual(interp_1d(2.25), 0.5)
            self.assertEqual(interp_1d(2.5), 1)
            self.assertEqual(interp_1d(2.75), 1.5)
            self.assertEqual(interp_1d(3), 2)
            self.assertEqual(interp_1d(5), 2)  # value outside domain
            self.assertEqual(interp_1d(0), -2)  # value outside domain

            assert_allclose(interp_1d((1, 1, 1)), (-2, -2, -2))
            assert_allclose(interp_1d((0, 1, 4)), (-2, -2, 2))  # some values outside domain
            assert_allclose(interp_1d(np.array((1, 1, 1))), (-2, -2, -2))  # numpy array

            assert_allclose(interp_1d(np.linspace(1, 3, 10)), np.linspace(-2, 2, 10))
            assert_allclose(interp_1d(()), ())  # empty table

    def test_1d_nel(self):
        logger.info("Testing data sampled from constant function")
        for kind in ('linear', 'spline'):
            interp_1d = RegularInterpolator(x=(1., 3., 4., 5., 10.), y=(2.0, 2.0, 2.0, 2.0, 2.0), kind=kind)
            assert_allclose(interp_1d(np.linspace(-1, 10, 100)), np.ones(100) * 2.0)

            interp_1d = RegularInterpolator(x=(1., 3., 4., 5.), y=(2.0, 2.0, 2.0, 2.0), kind=kind)
            assert_allclose(interp_1d(np.linspace(-1, 10, 100)), np.ones(100) * 2.0)

            interp_1d = RegularInterpolator(x=(1., 3., 4.), y=(2.0, 2.0, 2.0), kind=kind)
            assert_allclose(interp_1d(np.linspace(-1, 10, 100)), np.ones(100) * 2.0)

        logger.info("Testing some random data")
        data_x = (1., 3., 4., 5., 10.)
        data_y = (0.0, 10.0, 5.0, 0.0, -5.0)
        interp_1d = RegularInterpolator(x=data_x, y=data_y, kind='linear')
        assert_allclose(interp_1d(data_x), data_y)
        self.assertEqual(interp_1d(2), 5)
        self.assertEqual(interp_1d(2.5), 7.5)
        self.assertEqual(interp_1d(3.5), 7.5)
        self.assertEqual(interp_1d(7.5), -2.5)

        interp_1d = RegularInterpolator(x=data_x, y=data_y, kind='spline')
        assert_allclose(interp_1d(data_x), data_y, atol=1e-12)
        self.assertGreater(interp_1d(2), 10)  # spline interpolation making strange "hops"
        self.assertGreater(interp_1d(2.5), 10)
        assert_allclose(interp_1d(3.5), 7.5, atol=0.4)
        self.assertLess(interp_1d(7.5), -5)

        logger.info("Testing data sampled from identity function")
        data_y = data_x
        interp_1d = RegularInterpolator(x=data_x, y=data_y, kind='spline')
        assert_allclose(interp_1d(data_x), data_y, atol=1e-12)
        test_x = np.linspace(start=data_x[0], stop=data_x[-1], num=100)
        assert_allclose(interp_1d(test_x), test_x, atol=1e-12)

    def test_2d_2x2el(self):
        """
        Should fallback to linear interpolation
        """
        for kind in ('linear', 'spline'):
            interp_2d = RegularInterpolator(x=(1., 3.), y=(-2.0, 2.0), z=((2, 2), (3, 3)), kind=kind)
            self.assertEqual(interp_2d(1, -2), 2)
            self.assertEqual(interp_2d(1, 0), 2)
            self.assertEqual(interp_2d(1, 2), 2)

            self.assertEqual(interp_2d(3, -2), 3)
            self.assertEqual(interp_2d(3, 0), 3)
            self.assertEqual(interp_2d(3, 2), 3)

            self.assertEqual(interp_2d(2, -2), 2.5)
            self.assertEqual(interp_2d(2, 0), 2.5)
            self.assertEqual(interp_2d(2, 2), 2.5)

    def test_2d_2xnel(self):
        """
        Should partially fallback to linear interpolation
        """
        for kind in ('linear', 'spline'):
            interp_2d = RegularInterpolator(x=(1., 3., 10.), y=(-2.0, 2.0), z=((2, 2), (3, 3), (4, 4)), kind=kind)
            assert_allclose(interp_2d(1, -2), 2)
            assert_allclose(interp_2d(1, 0), 2)
            assert_allclose(interp_2d(1, 2), 2)

            assert_allclose(interp_2d(3, -2), 3)
            assert_allclose(interp_2d(3, 0), 3)
            assert_allclose(interp_2d(3, 2), 3)
            assert_allclose(interp_2d(x=(3, 3, 3), y=(-2, 0, 2)), (3, 3, 3))

            assert_allclose(interp_2d(2, -2), 2.5, atol=0.1)
            assert_allclose(interp_2d(2, 0), 2.5, atol=0.1)
            assert_allclose(interp_2d(2, 2), 2.5, atol=0.1)
            assert_allclose(RegularInterpolator.eval(x=2,
                                                     y=2,
                                                     xp=(1., 3., 10.),
                                                     yp=(-2.0, 2.0),
                                                     zp=((2, 2), (3, 3), (4, 4)),
                                                     kind=kind),
                            2.5,
                            atol=0.1)

            interp_2d = RegularInterpolator(x=(1., 3.), y=(-2.0, 2.0, 10.0), z=((2, 2, 2), (3, 3, 3)), kind=kind)
            assert_allclose(interp_2d(1, -2), 2)
            assert_allclose(interp_2d(1, 0), 2)
            assert_allclose(interp_2d(1, 2), 2)
            assert_allclose(interp_2d(1, 10), 2)

            assert_allclose(interp_2d(3, -2), 3)
            assert_allclose(interp_2d(3, 0), 3)
            assert_allclose(interp_2d(3, 2), 3)
            assert_allclose(interp_2d(3, 10), 3)

            assert_allclose(interp_2d(2, -2), 2.5, atol=0.1)
            assert_allclose(interp_2d(2, 0), 2.5, atol=0.1)
            assert_allclose(interp_2d(2, 2), 2.5, atol=0.1)
            assert_allclose(interp_2d(2, 10), 2.5, atol=0.1)

    def test_2d_nxnel(self):
        for kind in ('linear', 'spline'):
            interp_2d = RegularInterpolator(x=(1., 3., 10.),
                                            y=(-2.0, 0.0, 2.0),
                                            z=((2, 2, 2), (3, 3, 3), (4, 4, 4)),
                                            kind=kind)

            assert_allclose(interp_2d(1, -2), 2)
            assert_allclose(interp_2d(1, 0), 2)
            assert_allclose(interp_2d(1, 2), 2)

            assert_allclose(interp_2d(3, -2), 3)
            assert_allclose(interp_2d(3, 0), 3)
            assert_allclose(interp_2d(3, 2), 3)
            assert_allclose(interp_2d(x=(3, 3, 3), y=(-2, 0, 2)), (3, 3, 3))

            assert_allclose(interp_2d(2, -2), 2.5, atol=0.1)
            assert_allclose(interp_2d(2, 0), 2.5, atol=0.1)
            assert_allclose(interp_2d(2, 2), 2.5, atol=0.1)
            assert_allclose(RegularInterpolator.eval(x=2,
                                                     y=2,
                                                     xp=(1., 3., 10.),
                                                     yp=(-2.0, 2.0),
                                                     zp=((2, 2), (3, 3), (4, 4)),
                                                     kind=kind),
                            2.5,
                            atol=0.1)

            interp_2d = RegularInterpolator(x=(1., 3.), y=(-2.0, 2.0, 10.0), z=((2, 2, 2), (3, 3, 3)), kind=kind)
            assert_allclose(interp_2d(1, -2), 2)
            assert_allclose(interp_2d(1, 0), 2)
            assert_allclose(interp_2d(1, 2), 2)
            assert_allclose(interp_2d(1, 10), 2)

            assert_allclose(interp_2d(3, -2), 3)
            assert_allclose(interp_2d(3, 0), 3)
            assert_allclose(interp_2d(3, 2), 3)
            assert_allclose(interp_2d(3, 10), 3)

            assert_allclose(interp_2d(2, -2), 2.5, atol=0.1)
            assert_allclose(interp_2d(2, 0), 2.5, atol=0.1)
            assert_allclose(interp_2d(2, 2), 2.5, atol=0.1)
            assert_allclose(interp_2d(2, 10), 2.5, atol=0.1)


if __name__ == '__main__':
    unittest.main()
