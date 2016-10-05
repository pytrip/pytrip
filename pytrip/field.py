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
from math import log, sqrt, pi
from functools import cmp_to_key

import numpy as np

from pytrip.res.point import angles_from_trip, max_list, min_list
from pytrip.res.point import get_basis_from_angles
import pytriplib


def compare_raster_point(a, b):
    if a[1] is b[1]:
        return int(a[0] - b[0])
    return int(a[1] - b[1])


class Field:
    def __init__(self, ddd):
        self.ddd = ddd

    def get_cube_basis(self):
        return get_basis_from_angles(self.gantry_angle, self.couch_angle)

    def load_from_raster_points(self, rst):
        self.rst = rst
        self.bolus = rst.bolus
        self.couch_angle = float(rst.couchangle)
        self.gantry_angle = float(rst.gantryangle)
        self.gantry_angle, self.couch_angle = angles_from_trip(self.gantry_angle, self.couch_angle)
        self.subfields = []
        field_size = None
        for submachine in self.rst.get_submachines():
            sub = SubField(submachine, self.ddd, rst)
            self.subfields.append(sub)
            size = sub.get_size()
            if field_size is None:
                field_size = size
            else:
                field_size[0:5:2] = max_list(size[0:5:2], field_size[0:5:2])
                field_size[1:6:2] = min_list(size[1:6:2], field_size[1:6:2])
        self.field_size = field_size
        self.get_merged_raster_points()

    def get_merged_raster_points(self):
        if hasattr(self, "raster_matrix"):
            return self.raster_matrix
        size = None
        for sub in self.rst.get_submachines():
            tmp = sub.get_raster_min_max()
            if size is None:
                size = tmp
            else:
                size[0:3:2] = min_list(size[0:3:2], tmp[0:3:2])
                size[1:4:2] = max_list(size[1:4:2], tmp[1:4:2])
        matrixs = []
        for sub in self.subfields:
            matrixs.append(sub.get_merge_raster_points(size))
        self.raster_matrix = matrixs
        return np.array(matrixs)

    def get_energy_list(self):
        energy_list = []
        for sub in self.rst.get_submachines():
            energy_list.append(sub.energy)
        return energy_list

    def get_ddd_list(self):
        if hasattr(self, "ddd_list"):
            return self.ddd_list
        e_list = self.get_energy_list()
        self.ddd_list = self.ddd.get_ddd_grid(e_list, 1000)
        return self.ddd_list

    def get_max_dist(self):
        m = 0
        for submachine in self.rst.get_submachines():
            m = max(self.ddd.get_dist(submachine.energy), m)
        self.max_dist = m
        return m


class SubField:
    def __init__(self, submachine, ddd, rst, resolution=0.5):
        self.submachine = submachine
        self.sigma = self.submachine.focus
        self.energy = self.submachine.energy
        self.rst = rst
        self.ddd = ddd

    def get_lateral(self, resolution=0.5):
        if hasattr(self, "lateral") and resolution == resolution:
            return self.lateral
        max_dist = self.get_max_dist()
        dim = np.ceil(np.absolute(max_dist / resolution))
        max_dist = dim * resolution
        self.resolution = resolution
        a = np.meshgrid(np.linspace(0, max_dist, dim), np.linspace(0, max_dist, dim))
        r = (a[0]**2 + a[1]**2)**0.5
        sigma = self.sigma / ((8 * log(2))**0.5)
        lateral = 1 / ((2 * pi * sigma**2)**0.5) * np.exp(-(r**2) / (2 * sigma**2))
        tot_lat = np.zeros((2 * dim - 1, 2 * dim - 1))

        tot_lat[dim - 1:2 * dim - 1, dim - 1:2 * dim - 1] = lateral
        tot_lat[dim - 1:2 * dim - 1, 0:dim - 1] = np.rot90(lateral, 3)[:, 0:dim - 1]
        tot_lat[0:dim - 1, 0:dim - 1] = np.rot90(lateral, 2)[0:dim - 1, 0:dim - 1]
        tot_lat[0:dim - 1, dim - 1:2 * dim - 1] = np.rot90(lateral)[0:dim - 1, :]

        self.lateral = tot_lat
        return self.lateral

    def get_size(self):
        beamsize = self.get_max_dist()
        depth = self.ddd.get_dist(self.submachine.energy)
        lateral = self.submachine.get_raster_min_max()
        lateral[0] -= beamsize
        lateral[1] += beamsize
        lateral[2] -= beamsize
        lateral[3] += beamsize
        lateral.extend([0, depth])
        return lateral

    def get_merge_raster_points(self, size):
        points = np.array(self.get_raster_matrixs(size))
        dim = [len(points[0]), len(points)]

        points = np.reshape(points, (dim[0] * dim[1], 3))
        sigma = self.sigma / ((8 * log(2))**0.5)
        self.points = pytriplib.merge_raster_grid(np.array(points), sigma)
        self.points = np.reshape(self.points, (dim[1], dim[0], 3))
        return self.points

    def get_raster_matrixs(self, size):
        points = sorted(self.submachine.get_raster_points(), key=cmp_to_key(compare_raster_point))
        step = self.submachine.stepsize
        margin = 5
        mat = [
            [[x, y, 0]
             for x in np.linspace(size[0] - margin * step[0], size[1] + margin * step[0], (size[1] - size[0]
                                                                                           ) / step[0] + 1 + 2 * margin)
             ]
            for y in np.linspace(size[2] - margin * step[1], size[3] + margin * step[1], (size[3] - size[2]
                                                                                          ) / step[1] + 1 + 2 * margin)
        ]
        for p in points:
            i = int(p[1] / step[1] - size[2] / step[1]) + margin
            j = int(p[0] / step[0] - size[0] / step[0]) + margin
            mat[i][j] = p
        return mat

    def get_subfield_cube(self, resolution=0.5):

        size = self.get_size()
        # ~ doesn't allow minimum values greater than zero
        dim = np.ceil(np.absolute(np.array(size) / resolution))
        field = np.zeros((dim[5] + dim[4], dim[3] + dim[2] - 1, dim[1] + dim[0] - 1))
        lateral = self.get_lateral(resolution)
        ddd = self.ddd.get_ddd_by_energy(self.energy, np.linspace(0, (dim[5]) * resolution, dim[5] * 10))
        ddd = np.sum(np.reshape(ddd, (dim[5], -1)), axis=1) / 10

        raster_dim = [len(lateral[0]), len(lateral), len(ddd)]
        zero = np.ceil(np.array(self.zero) / resolution)
        raster_field = np.array([x * lateral for x in ddd])
        raster = self.submachine.get_raster_points()
        for point in raster:
            idx = np.array([point[0] / resolution, point[1] / resolution])
            raster_center = idx + zero
            start = np.array(
                [raster_center[0] - (raster_dim[0] + 1) / 2, raster_center[1] - (raster_dim[1] + 1) / 2, 0], dtype=int)
            end = np.array(
                [raster_center[0] + (raster_dim[0] - 1) / 2, raster_center[1] + (raster_dim[1] - 1) / 2, raster_dim[2]],
                dtype=int)
            field[start[2]:end[2], start[1]:end[1], start[0]:end[0]] += point[2] * raster_field
        return field, zero

    def get_max_dist(self):
        max_r = self.sigma * sqrt(log(100) / log(2))
        return max_r
