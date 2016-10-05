"""
    This file is part of PyTRiP.

    PyTRiP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyTRiP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyTRiP.  If not, see <http://www.gnu.org/licenses/>
"""

import os

import numpy as np
from pytrip.cube import Cube
import pytriplib


class LETCube(Cube):

    data_file_extension = "dos"

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
        size = np.array([self.pixel_size, self.pixel_size, self.slice_distance])
        lv = np.zeros(3000)
        for i in range(self.dimz):
            pos += self.slice_distance
            slice = voi.get_slice_at_pos(pos)
            if slice is not None:
                lv += pytriplib.calculate_lvh_slice(self.cube[i], np.array(slice.contour[0].contour), size)

        cubes = sum(lv)
        lvh = np.cumsum(lv[::-1])[::-1] / cubes
        min_let = np.where(lvh >= 0.98)[0][-1]
        max_let = np.where(lvh <= 0.02)[0][0]
        area = cubes * size[0] * size[1] * size[2] / 1000
        mean = np.dot(lv, range(0, 3000)) / cubes
        return lvh, min_let, max_let, mean, area

    def write_lvh_to_file(self, voi, path):
        lvh, _, _, _, _ = self.calculate_lvh(voi)
        print(lvh.shape)
        output = ""
        # TODO fix following line
        # lvh has shape (3000,0) and contains only let, not vol
        # see calculate_lvh() method
        for vol, let in zip(lvh[0], lvh[1]):
            output += "%.4e\t%.4e\n" % (vol, let)
        f = open(path + ".dvh", "w+")
        f.write(output)
        f.close()
