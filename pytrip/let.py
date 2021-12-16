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
This module provides the LETCube for handling LET data.
"""
import warnings

import numpy as np

from pytrip.cube import Cube


class LETCube(Cube):
    """ This class handles LETCubes.

    It is similar to DosCubes and CtxCubes, but is intended to hold LET data.
    The object has build-in methods to read and write the LET data cubes,
    calculate the LET-volume histograms, and write these to disk.
    It is inherited from Cube, which contains many additional methods and attributes.
    """

    data_file_extension = '.dos'  # skipcq: TYP-050
    allowed_suffix = ('dosemlet', 'mlet')  # skipcq: TYP-050

    let_types = {
        "unknown": "unknown LET type",
        "DLET": "dose-averaged LET",
        "DLET*": "dose-averaged LET, all secondaries",
        "DLETP": "dose-averaged LET, protons only",
        "TLET": "track-averaged LET",
        "TLET*": "track-averaged LET, all secondaries",
        "TLETP": "track-averaged LET, protons only"
    }

    def __init__(self, cube=None):
        super(LETCube, self).__init__(cube)
        self.type = "LET"
        self.let_type = None

    def __str__(self):
        if self.type == "LET":
            return "LET: " + self.basename
        return "LET (type " + self.type + "): " + self.basename

    def get_max(self):
        """ Returns the largest value in the LETCube.

        :returns: the largest value found in the in the LETCube.
        """
        return np.amax(self.cube)

    def calculate_lvh(self, voi):
        """ Calculate a LET-volume histogram.

        :param Voi voi: The volume of interest, in the form of a Voi object.
        :returns: A tuple containing
                  - lvh: the LET-volume histogram
                  - min_lvh: array of LET values below 2%
                  - max_lvh: array of LET values above 98%
                  - area: TODO - what is this?
        """
        warnings.warn("The method calculate_lvh() is deprecated, and is replaced with the pytrip.VolHist object.",
                      DeprecationWarning)
        from pytrip import pytriplib
        pos = 0
        size = np.array([self.pixel_size, self.pixel_size, self.slice_distance])
        lv = np.zeros(3000)
        for i in range(self.dimz):
            pos += self.slice_distance
            slice_at_pos = voi.get_slice_at_pos(pos)
            if slice_at_pos is not None:
                lv += pytriplib.calculate_lvh_slice(self.cube[i], np.array(slice_at_pos.contours[0].contour), size)

        cubes = sum(lv)
        lvh = np.cumsum(lv[::-1])[::-1] / cubes
        min_let = np.where(lvh >= 0.98)[0][-1]
        max_let = np.where(lvh <= 0.02)[0][0]
        area = cubes * size[0] * size[1] * size[2] / 1000
        mean = np.dot(lv, range(0, 3000)) / cubes
        return lvh, min_let, max_let, mean, area

    def write_lvh_to_file(self, voi, path):
        """ Write the LET-volume histogram to a file.

        :param Voi voi: The volume of interest, n the form of a Voi object.
        :param str path: Full path of file to be written.
        """
        lvh, _, _, _, _ = self.calculate_lvh(voi)
        print(lvh.shape)
        output = ""
        # TODO fix following line
        # lvh has shape (3000,0) and contains only let, not vol
        # see calculate_lvh() method
        for vol, let in zip(lvh[0], lvh[1]):
            output += "%.4e\t%.4e\n" % (vol, let)
        with open(path + ".dvh", "w+") as f:
            f.write(output)
