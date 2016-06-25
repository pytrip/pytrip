"""
    This file is part of PyTRiP.

    libdedx is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    libdedx is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with libdedx.  If not, see <http://www.gnu.org/licenses/>
"""
import numpy
import numpy as np
from pytrip.error import *
from pytrip.cube import *
import pytrip as plib

__author__ = "Niels Bassler and Jakob Toftegaard"
__version__ = "1.0"
__email__ = "bassler@phys.au.dk"


class LETCube(Cube):
    def __init__(self, cube=None):
        super(LETCube, self).__init__(cube)

    def write(self, path):
        f_split = os.path.splitext(path)
        header_file = f_split[0] + ".dosemlet.hed"
        dos_file = f_split[0] + ".dosemlet.dos"
        self.write_trip_header(header_file)
        self.write_trip_data(dos_file)

    def get_max(self):
        return np.amax(self.cube)

    def calculate_lvh(self, voi):
        pos = 0
        size = numpy.array([self.pixel_size, self.pixel_size, self.slice_distance])
        lv = numpy.zeros(3000)
        for i in range(self.dimz):
            pos += self.slice_distance
            slice = voi.get_slice_at_pos(pos)
            if slice is not None:
                lv += plib.calculate_lvh_slice(self.cube[i], numpy.array(slice.contour[0].contour), size)

        cubes = sum(lv)
        lvh = numpy.cumsum(lv[::-1])[::-1] / cubes
        min_let = numpy.where(lvh >= 0.98)[0][-1]
        max_let = numpy.where(lvh <= 0.02)[0][0]
        area = cubes * size[0] * size[1] * size[2] / 1000
        mean = numpy.dot(lv, range(0, 3000)) / cubes
        return (lvh, min_let, max_let, mean, area)

    def write_lvh_to_file(self, voi, path):
        lvh = self.calculate_lvh(voi)
        output = ""
        for vol, let in zip(lvh[0], lvh[1]):
            output += "%.4e\t%.4e\n" % (vol, let)
        f = open(path + ".dvh", "w+")
        f.write(output)
        f.close()
