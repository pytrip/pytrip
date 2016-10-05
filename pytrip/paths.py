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

import gc
import os
from math import sin, cos
from multiprocessing import Process, Queue
from functools import cmp_to_key

import numpy as np
from scipy import interpolate

from pytrip.cube import Cube
from pytrip.res.point import get_basis_from_angles
import pytriplib


def cmp_sort(a, b):
    if a["gantry"] == b["gantry"]:
        return a["couch"] - b["couch"]
    return a["gantry"] - b["gantry"]


class DensityProjections:
    def __init__(self, cube):
        self.cube = cube

    def calculate_quality_grid(self, voi, gantry, couch, calculate_from=0, stepsize=1.0, avoid=[], gradient=True):
        l = self.calculate_quality_list(voi, gantry, couch, calculate_from, stepsize, avoid=avoid, gradient=gradient)
        l = sorted(l, key=cmp_to_key(cmp_sort))
        grid_data = []
        for x in l:
            grid_data.append(x["data"][0])
        l = np.reshape(grid_data, (len(gantry), len(couch)))
        return l

    def calculate_quality_list(self, voi, gantry, couch, calculate_from=0, stepsize=1.0, avoid=[], gradient=True):
        q = Queue(32767)
        process = []
        d = voi.get_voi_cube()
        d.cube = np.array(d.cube, dtype=np.float32)
        voi_cube = DensityProjections(d)
        result = []
        for gantry_angle in gantry:
            p = Process(
                target=self.calculate_angle_quality_thread,
                args=(voi, gantry_angle, couch, calculate_from, stepsize, q, avoid, voi_cube, gradient))
            p.start()
            p.deamon = True
            process.append(p)
            if len(process) > 2:
                tmp = q.get()
                result.append(tmp)
                for p in process:
                    if not p.is_alive():
                        process.remove(p)
        while not len(result) == len(gantry) * len(couch):
            tmp = q.get()
            result.append(tmp)
        return result

    def calculate_best_angles(self, voi, fields, min_dist=20, gantry_limits=[-90, 90], couch_limits=[0, 359], avoid=[]):
        grid = self.calculate_quality_list(
            voi,
            range(gantry_limits[0], gantry_limits[1], 20),
            range(couch_limits[0], couch_limits[1], 20),
            avoid=avoid)
        grid = sorted(grid, key=lambda x: x["data"][0])
        best_angles = []
        i = 0
        while len(best_angles) < fields or i >= len(grid):
            is_ok = True
            point = np.array([grid[i]["gantry"], grid[i]["couch"]])
            for angle in best_angles:
                if np.linalg.norm(angle - point) < 2 * min_dist:
                    is_ok = False
            if is_ok:
                best_angles.append(self.optimize_angle(voi, point[0], point[1], 20, 3, avoid=avoid))
            i += 1
        return best_angles

    def optimize_angle(self, voi, couch, gantry, margin, iteration, avoid=[]):
        if iteration is 0:
            return [gantry, couch]
        grid = self.calculate_quality_list(
            voi, np.linspace(gantry, gantry + margin, 3), np.linspace(couch, couch + margin, 3), avoid=avoid)
        min_item = min(grid, key=lambda x: x["data"][0])
        return self.optimize_angle(voi, min_item["gantry"], min_item["couch"], margin / 2, iteration - 1, avoid=avoid)

    def calculate_angle_quality_thread(self,
                                       voi,
                                       gantry,
                                       couch,
                                       calculate_from=0,
                                       stepsize=1.0,
                                       q=None,
                                       avoid=[],
                                       voi_cube=None,
                                       gradient=True):
        os.nice(1)
        for couch_angle in couch:
            qual = self.calculate_angle_quality(voi, gantry, couch_angle, calculate_from, stepsize, avoid, voi_cube,
                                                gradient)
            q.put({"couch": couch_angle, "gantry": gantry, "data": qual})

    def calculate_angle_quality(self,
                                voi,
                                gantry,
                                couch,
                                calculate_from=0,
                                stepsize=1.0,
                                avoid=[],
                                voi_cube=None,
                                gradient=True):
        voi_min, voi_max = voi.get_min_max()
        for v in avoid:
            v_min, v_max = v.get_min_max()
        if voi_cube is None:
            d = voi.get_voi_cube()
            d.cube = np.array(d.cube, dtype=np.float32)
            voi_cube = DensityProjections(d)

        data, start, basis = self.calculate_projection(voi, gantry, couch, calculate_from, stepsize)
        voi_proj, t1, t2 = voi_cube.calculate_projection(voi, gantry, couch, 1, stepsize)

        if gradient:
            gradient = np.gradient(data)
            data = (gradient[0]**2 + gradient[1]**2)**0.5
        a = data * (voi_proj > 0.0)
        quality = sum(a)
        area = sum(voi_proj > 0.0)
        # ~ area = sum(data>0.0)/10
        if gradient:
            mean_quality = 10 - abs(quality / area)
        else:
            mean_quality = abs(quality / area)
        return mean_quality, quality, area

    def calculate_projection(self, voi, gantry, couch, calculate_from=0, stepsize=1.0):
        """
        :param voi: tumortarget, type is Voi
        :param gantry: angle in degree
        :param couch: angle in degree
        :param calculate_from: 0 is mass center 1 is the most distance point
        in the tumor from the beamaxis
        :param stepsize: relative to pixelsize 1 is a step of 1 pixel
        :return:
        """
        # min_structure = 5 TODO why not used ?
        basis = get_basis_from_angles(gantry, couch)
        # Convert angles from degrees to radians
        gantry /= 180.0 / np.pi
        couch /= 180.0 / np.pi
        # calculate surface normal
        step_vec = -stepsize * self.cube.pixel_size * np.array(basis[0])
        step_length = stepsize * self.cube.pixel_size
        center = voi.calculate_center()
        # ~ (b,c) = self.calculate_plane_vectors(gantry,couch)
        b = basis[1]
        c = basis[2]
        min_window, max_window = voi.get_min_max()
        size = np.array(max_window) - np.array(min_window)
        window_size = np.array([((sin(couch) * size[1])**2 + (cos(couch) * size[2])**2)**0.5, (
            (sin(gantry) * size[0])**2 + (cos(gantry) * size[1])**2 + (sin(couch) * size[2])**2)**0.5]) * 2

        dimension = window_size / self.cube.pixel_size
        dimension = np.int16(dimension)
        start = center - self.cube.pixel_size * 0.5 * np.array(dimension[0] * b + dimension[1] * c)
        if calculate_from is 1:
            start = self.calculate_back_start_voi(voi, start, step_vec)
        if calculate_from is 2:
            start = self.calculate_front_start_voi(voi, start, step_vec)

        data = pytriplib.calculate_wepl(
            self.cube.cube, np.array(start), np.array(basis) * step_length, dimension,
            np.array([self.cube.pixel_size, self.cube.pixel_size, self.cube.slice_distance]))
        data *= step_length
        return data, start, [b, c]

    def calculate_back_start_voi(self, voi, start, beam_axis):
        points = voi.get_3d_polygon() - start
        distance = min(np.dot(points, beam_axis))
        return start + distance * beam_axis

    def calculate_front_start_voi(self, voi, start, beam_axis):
        points = voi.get_3d_polygon() - start
        distance = max(np.dot(points, beam_axis))
        return start + distance * beam_axis


class DensityCube(Cube):
    def __init__(self, ctxcube):
        self.ctxcube = ctxcube
        super(DensityCube, self).__init__(ctxcube)
        self.type = "Density"
        self.directory = os.path.dirname(os.path.abspath(__file__))
        self.hlut_file = self.directory + "/data/hlut_den.dat"
        self.import_hlut()
        self.calculate_cube()
        self.ctxcube = None

    def calculate_cube(self):
        ctxdata = self.ctxcube.cube
        ctxdata = ctxdata.reshape(self.dimx * self.dimy * self.dimz)
        # ~ print self.dimx*self.dimy*self.dimz/1000000
        gc.collect()

        cube = interpolate.splev(ctxdata, self.hlut_data)
        # ~ cube = self.hlut_data(ctxdata)
        cube = cube.astype(np.float32)
        self.cube = np.reshape(cube, (self.dimz, self.dimy, self.dimx))

    def import_hlut(self):
        fp = open(self.hlut_file, "r")
        lines = fp.read()
        fp.close()
        lines = lines.split('\n')
        x_data = []
        y_data = []
        for line in lines:
            a = line.split()
            if len(a):
                x_data.append(float(a[0]))
                y_data.append(float(a[3]))
        self.hlut_data = interpolate.splrep(np.array(x_data), np.array(y_data), k=1)
