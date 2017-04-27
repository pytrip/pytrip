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
Module with auxilliary functions (mostly internal use).
"""

import numpy as np


def get_class_name(item):
    """
    :returns: name of class of 'item' object.
    """
    return item.__class__.__name__


def evaluator(funct, name='funct'):
    """ Wrapper for evaluating a function.

    :params str funct: string which will be parsed
    :params str name: name which will be assigned to created function.

    :returns: function f build from 'funct' input.
    """
    code = compile(funct, name, 'eval')

    def f(x):
        return eval(code, locals())

    f.__name__ = name
    return f


def volume_histogram(cube, voi=None, bins=256):
    """
    Generic volume histogram calculator, useful for DVH and LVH or similar.

    :params cube: a data cube of any shape, e.g. Dos.cube
    :params voi: optional voi where histogramming will happen.
    :returns [x],[y]: coordinates ready for plotting. Dose (or LET) along x, Normalized volume along y in %.

    If VOI is not given, it will calculate the histogram for the entire dose cube.

    Providing voi will slow down this function a lot, so if in a loop, it is recommended to do masking
    i.e. only provide Dos.cube[mask] instead.
    """

    if voi is None:
        mask = None
    else:
        vcube = voi.get_voi_cube()
        mask = (vcube.cube == 1000)

    _xrange = (0.0, cube.max()*1.1)
    _hist, x = np.histogram(cube[mask], bins=bins, range=_xrange)
    _fhist = _hist[::-1]  # reverse historgram, so first element is for highest dose
    _fhist = np.cumsum(_fhist)
    _hist = _fhist[::-1]  # flip back again to normal representation

    y = 100.0 * _hist / _hist[0]  # volume histograms always plot the right edge of bin, since V(D < x_pos).
    y = np.insert(y, 0, 100.0, axis=0)  # but the leading bin edge is always at V = 100.0%

    return x, y
