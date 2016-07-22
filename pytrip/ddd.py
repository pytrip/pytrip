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
import glob
import numpy as np
from scipy import interpolate


class DDD:
    def __init__(self):
        pass

    def get_ddd_data(self, energy, points):
        e = self.get_nearest_energy(energy)
        return interpolate.splev(points, self.ddd_data[e])

    def get_dist(self, energy):
        return interpolate.splev(energy, self.max_dist)

    def get_ddd_by_energy(self, energy, points):
        ev_point = np.array([points, [energy] * len(points)])
        return interpolate.griddata(self.points, self.ddd_list, np.transpose(ev_point), method='linear')

    def get_ddd_grid(self, energy_list, n):
        # ddd_list = []
        energy = []
        dist = []
        # point = []
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
            spl = interpolate.splrep(depth, ddd)
            data.extend(interpolate.splev(xi, spl))
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

    def load_ddd(self, folderpath):
        x_data = []
        y_data = []
        points = [[], []]
        ddd = {}
        max_dist = []
        items = glob.glob(folderpath)
        for item in items:
            x_data = []
            y_data = []
            f = open(item)
            data = f.read()
            f.close()
            lines = data.split('\n')
            for n, line in enumerate(data.split("\n")):
                if line.find("energy") is not -1:
                    energy = float(line.split()[1])
                if line.find('!') is -1 and line.find('#') is -1:
                    break
            for i in range(n, len(lines)):
                if len(lines[i]) < 3:
                    continue
                point = [float(s) for s in lines[i].split()]
                x_data.append(point[0] * 10)
                y_data.append(point[1])
            ddd[energy] = np.array([x_data, y_data])
            max_dist.append([energy, x_data[-1]])

        max_dist = np.array(sorted(max_dist, key=lambda x: x[0]))
        self.max_dist = interpolate.splrep(max_dist[:, 0], max_dist[:, 1], s=0)
        #        spl_points = 1000
        #        xi = np.linspace(0, max_dist[-1, 1], spl_points)
        ddd_list = []
        for key, value in ddd.iteritems():
            points[0].extend(value[0])
            points[1].extend(len(value[0]) * [key])
            ddd_list.extend(value[1])
        self.ddd_list = ddd_list
        self.ddd = ddd
        self.points = np.transpose(points)
