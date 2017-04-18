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
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RegularInterpolator(object):
    """
    RegularInterpolator is a helper class to easy interpolation of single- and double-variable function.

    """

    def __init__(self, x, y, z=None, kind='spline'):
        self._executor = None

        # 1-D interpolation
        if z is None:
            self._executor = self.__get_1d_executor(x, y, kind)

        # 2-D interpolation
        else:
            # 2-D data, but one of the coordinates is single-element, reducing to 1-D interpolation
            if len(x) == 1:  # data with shape 1 x N
                self._executor = self.__get_1d_executor(x=y, y=z, kind=kind)
            elif len(y) == 1:  # data with shape N x 1
                self._executor = self.__get_1d_executor(x=x, y=z, kind=kind)
            else:
                # 3-rd degree spline interpolation, passing through all points
                self._executor = self.__get_2d_executor(x, y, z, kind)

    def __call__(self, x, y=None):
        if y is None:
            return self._executor(x)
        else:
            return self._executor(x, y)

    @staticmethod
    def __get_1d_executor(x, y, kind):

        def fun_interp(t):
            return np.interp(t, x, y)

        # input consistency check
        if len(x) != len(y):
            logger.error("len(x) = {:d}, len(y) = {:d}. Both should be equal".format(len(x), len(y)))
            raise Exception("1-D interpolation: X and Y should have the same shape")
        # 1-element data set, return fixed value
        if len(y) == 1:
            def fun_const(t):
                return y[0]
            result = fun_const
        # 2-element data set, use linear interpolation from numpy
        elif len(y) == 2:
            result = fun_interp
        else:
            if kind == 'spline':
                # 3-rd degree spline interpolation, passing through all points
                try:
                    from scipy.interpolate import InterpolatedUnivariateSpline
                except ImportError as e:
                    logger.error("Please install scipy for you platform to be able to use spline-based interpolation")
                    raise e
                result = InterpolatedUnivariateSpline(x, y, k=3)
            elif kind == 'linear':
                result = fun_interp
            else:
                raise ("Unsupported interpolation type {:s}.".format(kind))
        return result

    @staticmethod
    def __get_2d_executor(x, y, z, kind):
        try:
            from scipy.interpolate import RectBivariateSpline
        except ImportError as e:
            logger.error("Please install scipy for you platform to be able to use spline-based interpolation")
            raise e
        if kind == 'linear' or (len(x) == 2 and len(y) == 2):
            kx, ky = 1, 1
        elif len(x) == 2 and len(y) > 2:
            kx, ky = 1, 3
        elif len(x) > 2 and len(y) == 2:
            kx, ky = 3, 1
        else:
            kx, ky = 3, 3
        result = RectBivariateSpline(x, y, z, kx=kx, ky=ky, s=0)
        return result
