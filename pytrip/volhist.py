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
This module provides the volume histogram class.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VolHist:
    """
    Volume histogram class
    """
    def __init__(self, cube, voi=None, target_dose=None):
        """
        :params Cube cube: either LETCube, DosCube or similar object.
        :params Voi voi: a single voi (vdx.vois[i])
        :target_dose: set target_dose in [Gy]. Any target_dose in cube.target_dose will be ignored, if set.
        """

        self.cube_basename = cube.basename  # basename of the cube used for histogram
        if voi:
            self.name = voi.name  # name of the VOI
        else:
            self.name = "(none)"
        self.target_dose = cube.target_dose  # optional target dose scaling factor

        logger.info("Processing ROI '{:s}' for '{}'...".format(self.name, self.cube_basename))
        self.x, self.y = self.volume_histogram(cube.cube, voi)  # x,y data

        self.ylabel = "Volume [%]"  # units on y-axis

        self.x_is_relative = False  # relative or absolute x units

        if cube.type == 'DOS':  # DOS Cube
            # DOS cube data are stored in %%.
            if target_dose:
                _tdose = 0.001 * target_dose
                self.xlabel = "Dose [Gy]"  # units on x-axis
            elif self.target_dose > 0.0:
                _tdose = 0.001 * self.target_dose
                self.xlabel = "Dose [Gy]"
            else:
                _tdose = 0.1
                self.xlabel = "Dose [%]"
                self.x_is_relative = True
            logger.debug("Target dose {}".format(_tdose))
            self.x *= _tdose

        elif cube.type == 'LET':  # LET Cube
            self.xlabel = "LET [keV/um]"

        else:  # Unknown Cube
            self.xlabel = "(unkown)"

    @staticmethod
    def volume_histogram(data_cube, voi=None, bins=256):
        """
        Generic volume histogram calculator, useful for DVH and LVH or similar.

        :params data_cube: a data cube of any shape, e.g. Dos.cube
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

        _xrange = (0.0, data_cube.max() * 1.1)
        _hist, x = np.histogram(data_cube[mask], bins=bins, range=_xrange)
        _fhist = _hist[::-1]  # reverse histogram, so first element is for highest dose
        _fhist = np.cumsum(_fhist)
        _hist = _fhist[::-1]  # flip back again to normal representation

        y = 100.0 * _hist / _hist[0]  # volume histograms always plot the right edge of bin, since V(D < x_pos).
        y = np.insert(y, 0, 100.0, axis=0)  # but the leading bin edge is always at V = 100.0%

        return x, y
