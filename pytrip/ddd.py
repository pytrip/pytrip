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
This module provides the DDD class for handling depth-dose curve kernels for TRiP98.
"""
import glob
import logging
import numpy as np
from pytrip.res.interpolate import RegularInterpolator

logger = logging.getLogger(__name__)


class DDD:
    """ Class for handling Depth-Dose Data.
    """

    def get_ddd_data(self, energy, points):
        """ TODO: documentation
        """
        e = self.get_nearest_energy(energy)
        return self.ddd_data[e](points)

    def get_dist(self, energy):
        """ TODO: documentation
        """
        return self.max_dist(energy)

    def get_ddd_by_energy(self, energy, points):
        """ TODO: documentation
        """
        try:
            from scipy import interpolate
        except ImportError as e:
            logger.error("Please install scipy to be able to use spline-based interpolation")
            raise e
        ev_point = np.array([points, [energy] * len(points)])
        return interpolate.griddata(self.points, self.ddd_list, np.transpose(ev_point), method='linear')

    def get_ddd_grid(self, energy_list, n):
        """ TODO: documentation
        """
        energy = []
        dist = []
        data = []

        ddd_e = self.ddd.keys()
        ddd_e = sorted(ddd_e)

        for e in energy_list:
            idx = np.where((np.array(ddd_e) >= e))[0][0] - 1

            d_lower = self.ddd[ddd_e[idx]]
            d_upper = self.ddd[ddd_e[idx + 1]]

            lower_idx = np.where(max(d_lower[1, :]) == d_lower[1, :])[0][0]
            upper_idx = np.where(max(d_upper[1, :]) == d_upper[1, :])[0][0]

            offset = 1 / (ddd_e[idx + 1] - ddd_e[idx]) * (e - ddd_e[idx + 1])
            x_offset = (d_upper[0, upper_idx] - d_lower[0, lower_idx]) * offset
            y_offset = 1 + (1 - d_upper[1, upper_idx] / d_lower[1, lower_idx]) * offset

            depth = d_upper[0, :] + x_offset
            ddd = d_upper[1, :] * y_offset
            xi = np.linspace(0, depth[-1], n)
            spl = RegularInterpolator(x=depth, y=ddd)
            data.extend(spl(xi))
            dist.extend(xi)
            energy.extend([e] * n)

        out = [dist, energy, data]
        return np.reshape(np.transpose(out), (len(energy_list), n, 3))

        # TODO why it is not used ?
        # ddd_list = []
        # energy = []
        # dist = []
        # point = []
        # for e in energy_list:
        #     dist.extend(np.linspace(0, self.get_dist(e), n))
        #     energy.extend([e] * n)
        # point.append(dist)
        # point.append(energy)
        # data = interpolate.griddata(self.points,
        #                             self.ddd_list,
        #                             np.transpose(point),
        #                             method='linear')
        # out = [dist, energy, data]
        # return np.reshape(np.transpose(out), (len(energy_list), n, 3))

    def load_ddd(self, directory):
        """ Loads all .ddd files found in 'directory'

        :params str directory: directory where the .ddd files are found.
        """
        x_data = []
        y_data = []
        points = [[], []]
        ddd = {}
        max_dist = []
        items = glob.glob(directory)
        for item in items:
            x_data = []
            y_data = []
            with open(item, 'r') as f:
                data = f.read()
            lines = data.split('\n')
            n = 0
            for line in lines:
                if line.find("energy") != -1:
                    energy = float(line.split()[1])
                if line.find('!') == -1 and line.find('#') == -1:
                    break
                n += 1
            for i in range(n, len(lines)):
                if len(lines[i]) < 3:
                    continue
                point = [float(s) for s in lines[i].split()]
                x_data.append(point[0] * 10)
                y_data.append(point[1])
            ddd[energy] = np.array([x_data, y_data])
            max_dist.append([energy, x_data[-1]])

        max_dist = np.array(sorted(max_dist, key=lambda x: x[0]))
        self.max_dist = RegularInterpolator(x=max_dist[:, 0], y=max_dist[:, 1])
        ddd_list = []
        for key, value in ddd.items():
            points[0].extend(value[0])
            points[1].extend(len(value[0]) * [key])
            ddd_list.extend(value[1])
        self.ddd_list = ddd_list
        self.ddd = ddd
        self.points = np.transpose(points)
