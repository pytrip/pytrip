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
This package provides an easy way to perform interpolation of 1-D and 2-D datasets.

1-D dataset is represented by two columns of numbers: X and Y, of the same length.
X column should contain increasing sequence of numbers.
It is not required that X data should form regular (evenly spaced) grid.
X and Y columns of length 1 also form 1-D data (single data point).

2-D dataset is represented by two columns of numbers: X and Y.
and 2-dimensional matrix Z.
Number of columns in Z matrix should be equal to length of X array,
while number of rows to length of Y array.
Lengths of X and Y data may differ. One or both of X and Y columns may
contain single element - this is also valid 2-D dataset.
X and Y column should contain increasing sequence of numbers.
It is not required that X and Y data should form regular (evenly spaced) grid.

Two types of interpolation are supported: linear and spline (3rd degree polynomials).

Linear interpolation on 1-D dataset and on reduced 2-D datasets (where X or Y is single element column)
is performed using `interp` method from `numpy` package.

Spline interpolation on all datasets is performed using `InterpolatedUnivariateSpline` (for 1-D data)
and `RectBivariateSpline` (for 2-D data) from `scipy` package.

Cubic spline interpolation requires at least 4 data points. In case number of data points is lower,
interpolator implemented in this package will fall back to quadratic spline (for 3 data points)
or to linear interpolation (2 or less data points).
This is different behaviour from what is implemented in `scipy` package where exception is thrown is such situation.

Note: scipy package is not imported her as default as its installation is problematic for some group of users
(i.e. Windows users working without Anaconda python distribution).
Most of the functionality of this package (i.e. 1-D and some cases of 2-D linear interpolation) will work without
scipy package installed. If user without scipy package calls cubic spline interpolation method then an exception
will be thrown and an error message printed, requesting user to install scipy.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RegularInterpolator(object):
    """
    RegularInterpolator is a helper class to easy interpolation of single- and double-variable function.
    To use it usually two steps are needed:
     1. construction of the object, providing training data (data points) and interpolation type (linear or spline).
     During this step coefficient of interpolating function will be calculated and stored in the object.
     RegularInterpolator objects are so-called callables and can be called in same way as plain functions
     on interpolated values.
     Example:

         >> interp_func_1d = RegularInterpolator(x=exp_data_x, y=exp_data_y, kind='linear')
         >> interp_func_2d = RegularInterpolator(x=exp_data_x, y=exp_data_y, z=exp_data_z, kind='spline')

    2. Calling interpolation function to get intermediate values (single number or array) on interpolated ones.

         >> interpolated_y = interp_func_1d(x=intermediate_x)
         >> interpolated_z = interp_func_2d(x=intermediate_x, y=intermediate_y)

    Interpolation in single step is also possible (although less efficient than two-step method):

         >> interpolated_y = RegularInterpolator.eval(x=intermediate_x, xp=exp_data_x, yp=exp_data_y, kind='linear')
         >> interpolated_z = RegularInterpolator(x=intermediate_x, y=intermediate_y, \
        xp=exp_data_x, yp=exp_data_y, zp=exp_data_z, kind='spline')

    """

    def __init__(self, x, y, z=None, kind='spline'):
        """
        Initialisation responsible also for finding coefficients of interpolation function.
        If z is None then we assume that dataset if 1-D and is made by X and Y columns.
        In case z is not None, then we interpret the data as 2-D.

        :param x: array_like
        The 1-d array of data-points x-coordinates, must be in strictly ascending order.

        :param y: array_like
        The 1-d array of data-points y-coordinates.
        If y is from 1-D data points (z is None) then it should be of the same length (shape) as x.
        Otherwise (2-D data points, z is not None) then it must be in strictly ascending order.

        :param z: array_like
        None in case of 1-D dataset.
        Otherwise the 2-d array of data-points z-coordinates, of shape (x.size, y.size).

        :param kind: interpolation algorithm: 'linear' or 'spline' (default).
        """

        # internal executor object
        self._interp_function = None

        # 1-D interpolation
        if z is None:
            self._interp_function = self.__get_1d_function(x, y, kind)

        # 2-D interpolation
        else:
            # 2-D data, but one of the coordinates is single-element, reducing to 1-D interpolation
            if len(x) == 1:  # data with shape 1 x N
                executor_1d = self.__get_1d_function(x=y, y=z, kind=kind)

                def applicator(_unused, y, *args, **kwargs):
                    return executor_1d(y)
                self._interp_function = applicator
            elif len(y) == 1:  # data with shape N x 1
                executor_1d = self.__get_1d_function(x=x, y=z, kind=kind)

                def applicator(x, _unused, *args, **kwargs):
                    return executor_1d(x)
                self._interp_function = applicator
            else:
                # 3-rd degree spline interpolation, passing through all points
                self._interp_function = self.__get_2d_function(x, y, z, kind)

    def __call__(self, x, y=None):
        """
        Call interpolation function on intermediate values.

        :param x: array_like or single value
        Input x-coordinate(s).
        :param y: array_like or single value
        Input y-coordinate(s) (in case of 2-D dataset).
        :return: array_like or single value
        Interpolated value
        """
        if y is None:
            return self._interp_function(x)
        return self._interp_function(x, y, grid=False)

    @classmethod
    def eval(cls, x, y=None, xp=None, yp=None, zp=None, kind='linear'):
        """
        Perform interpolation in a single step. Find interpolation function,
         based on data points (xp, yp and zp) and then execute it on interpolated values (x and y).
        :param x: array_like or single value
        Input x-coordinate(s).

        :param y: array_like or single value
        Input y-coordinate(s) (in case of 2-D dataset).

        :param xp: array_like
        The 1-d array of data-points x-coordinates, must be in strictly ascending order.

        :param yp: array_like
        The 1-d array of data-points y-coordinates.
        If y is from 1-D data points (z is None) then it should be of the same length (shape) as x.
        Otherwise (2-D data points, z is not None) then it must be in strictly ascending order.

        :param zp: array_like
        None in case of 1-D dataset.
        Otherwise the 2-d array of data-points z-coordinates, of shape (x.size, y.size).

        :param kind: interpolation algorithm: 'linear' or 'spline' (default).
        :return: array_like or single value
        Interpolated value
        """
        if xp is None or yp is None:
            logger.error("Provide valid training data points")
            raise Exception("Invalid training data points")
        interpolating_class = cls(xp, yp, zp, kind)  # find interpolating function
        return interpolating_class(x, y)  # call interpolating function

    @staticmethod
    def __get_1d_function(x, y, kind):
        """
        Train 1-D interpolator
        :param x: x-coordinates of data points
        :param y: y-coordinates of data points
        :param kind: 'linear' or 'spline' interpolation type
        :return Interpolator callable object
        """
        def fun_interp(t):
            return np.interp(t, x, y)

        # input consistency check
        if len(x) != len(y):
            logger.error("len(x) = {:d}, len(y) = {:d}. Both should be equal".format(len(x), len(y)))
            raise Exception("1-D interpolation: X and Y should have the same shape")
        # 1-element data set, return fixed value
        if len(y) == 1:
            # define constant
            def fun_const(t):
                """
                Helper function
                :param t: array-like or scalar
                :return: array of constant values if t is an array of constant scalar if t is a scalar
                """
                try:
                    result = np.ones_like(t) * y[0]  # t is an array
                except TypeError:
                    result = y[0]  # t is a scalar
                return result
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
                    logger.error("Please install scipy on your platform to be able to use spline-based interpolation")
                    raise e
                k = 3
                if len(y) == 3:  # fall back to 2-nd degree spline if only 3 points are present
                    k = 2
                result = InterpolatedUnivariateSpline(x, y, k=k)
            elif kind == 'linear':
                result = fun_interp
            else:
                raise ValueError("Unsupported interpolation type {:s}.".format(kind))
        return result

    @staticmethod
    def __get_2d_function(x, y, z, kind):
        """
        Train 2-D interpolator
        :param x: x-coordinates of data points
        :param y: y-coordinates of data points
        :param z: z-coordinates of data points
        :param kind: 'linear' or 'spline' interpolation type
        :return Interpolator callable object
        """
        try:
            from scipy.interpolate import RectBivariateSpline
        except ImportError as e:
            logger.error("Please install scipy on your platform to be able to use spline-based interpolation")
            raise e
        if len(x) == 2:  # fall-back to linear interpolation
            kx = 1
        elif len(x) == 3:  # fall-back to 2nd degree spline
            kx = 2
        else:
            kx = 3
        if len(y) == 2:  # fall-back to linear interpolation
            ky = 1
        elif len(y) == 3:  # fall-back to 2nd degree spline
            ky = 2
        else:
            ky = 3
        if kind == 'linear':
            kx, ky = 1, 1
        x_array, y_array, z_array = x, y, z
        if not isinstance(x, np.ndarray):
            x_array = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y_array = np.asarray(y)
        if not isinstance(z, np.ndarray):
            z_array = np.asarray(z)
        result = RectBivariateSpline(x_array, y_array, z_array, kx=kx, ky=ky, s=0)
        return result
