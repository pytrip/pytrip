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
import numpy as np
import dicom
from pytrip.file_parser import parse_to_var


class Rst:
    def __init__(self):
        self.var_dict = {"rstfile": "rstfile",
                         "sistable": "sistable",
                         "patient_id": "patient_name",
                         "projectile": "projectile",
                         "gantryangle": "gantryangle",
                         "couchangle": "couchangle",
                         "#submachines": "submachines",
                         "bolus": "bolus",
                         "ripplefilter": "ripplefilter",
                         "mass": "mass",
                         "charge": "charge",
                         "#particles": "particles"}
        self.machines = []

    def get_submachines(self):
        return self.machines

    def calculate_total_energi(self):
        return

    def load_dicom(self, path):
        self.data = dicom.read_file(path)

    def load_field(self, path):
        with open(path, mode='r') as f:
            data = f.read()
        data = data.split("\n")
        out, i = parse_to_var(data, self.var_dict, "submachine#")
        for key, item in out.iteritems():
            setattr(self, key, item)
        if hasattr(self, "bolus"):
            self.bolus = float(self.bolus)
        for machine in range(int(self.submachines)):
            submachine = SubMachine()
            i = submachine.load_submachine(data, i)
            self.machines.append(submachine)

    def get_stepsize(self):
        if len(self.machines) > 0:
            return self.machines[0].stepsize
        return None

    def get_min_max(self):
        min_x = None
        max_x = None
        min_y = None
        max_y = None
        for submachine in self.machines:
            min_max = submachine.get_raster_min_max()
            if min_x is None or min_x > min_max[0]:
                min_x = min_max[0]
            if max_x is None or max_x > min_max[1]:
                max_x = min_max[1]
            if min_y is None or min_y > min_max[2]:
                min_y = min_max[2]
            if max_y is None or max_y > min_max[3]:
                max_y = min_max[3]
        return [min_x, max_x, min_y, max_y]

    def save_random_error_rst(self, path, sigma):
        # TODO why out here, not being used
        # out = self.generate_random_error_rst(sigma)

        with open(path, "wb+") as fp:
            fp.write("rstfile %s\n" % getattr(self, "rstfile"))
            fp.write("sistable %s\n" % getattr(self, "sistable"))
            fp.write("patient_id %s\n" % getattr(self, "patient_name"))
            fp.write("machine# 0\n")
            fp.write("projectile %s\n" % getattr(self, "projectile"))
            fp.write("charge %s\n" % getattr(self, "charge"))
            fp.write("mass %s\n" % getattr(self, "mass"))
            fp.write("gantryangle %s\n" % getattr(self, "gantryangle"))
            fp.write("couchangle %s\n" % getattr(self, "couchangle"))
            fp.write("stereotacticcoordinates\n")
            fp.write("bolus %.0f\n" % getattr(self, "bolus"))
            fp.write("ripplefilter %s\n" % getattr(self, "ripplefilter"))
            fp.write("#submachines %s\n" % getattr(self, "submachines"))
            fp.write("#particles %s\n" % getattr(self, "particles"))
            for machine in self.machines:
                machine.save_random_error_machine(fp, sigma)

    def generate_random_error_rst(self, sigma):
        out = []
        for submachine in self.machines:
            out.append(submachine.generate_random_error_machine(sigma))


class SubMachine:
    def __init__(self):
        self.raster_points = []

    def get_raster_points(self):
        return self.raster_points

    def get_raster_min_max(self):
        min_x = min(self.raster_points, key=lambda x: x[0])[0]
        min_y = min(self.raster_points, key=lambda x: x[1])[1]
        max_x = max(self.raster_points, key=lambda x: x[0])[0]
        max_y = max(self.raster_points, key=lambda x: x[1])[1]
        return [min_x, max_x, min_y, max_y]

    def get_raster_grid(self):
        min_max = self.get_raster_min_max()
        zero = [-int(min_max[0] / self.stepsize[0]), -int(min_max[2] / self.stepsize[1])]
        size = [(min_max[1] - min_max[0]) / self.stepsize[0] + 1, (min_max[3] - min_max[2]) / self.stepsize[1] + 1]
        grid = np.zeros(size)
        rasterpoints = np.array(self.raster_points)
        grid[np.array(rasterpoints[:, 0] / self.stepsize[0] + zero[0], 'uint8'), np.array(
            rasterpoints[:, 1] / self.stepsize[1] + zero[1], 'uint8')] = rasterpoints[:, 2]
        return grid

    def generate_random_error_machine(self, sigma):
        min_max = self.get_raster_min_max()
        zero = [-int(min_max[0] / self.stepsize[0]), -int(min_max[2] / self.stepsize[1])]
        size = [(min_max[1] - min_max[0]) / self.stepsize[0] + 1, (min_max[3] - min_max[2]) / self.stepsize[1] + 1]
        grid = np.zeros((size))
        rasterpoints = np.array(self.raster_points)
        grid[np.array(rasterpoints[:, 0] / self.stepsize[0] + zero[0], 'uint8'), np.array(
            rasterpoints[:, 1] / self.stepsize[1] + zero[1], 'uint8')] = rasterpoints[:, 2]
        size2 = [size[0] + 4 * int(sigma / self.stepsize[0]), size[1] + 4 * int(sigma / self.stepsize[1])]
        zero2 = [zero[0] + 2 * int(sigma / self.stepsize[0]), zero[1] + 2 * int(sigma / self.stepsize[1])]
        outgrid = np.zeros(size2)
        offset = np.array([[x, y]
                           for x in range(-int(sigma / self.stepsize[0]), int(sigma / self.stepsize[0]) + 1)
                           for y in range(-int(sigma / self.stepsize[1]), int(sigma / self.stepsize[1]) + 1)])
        lengths = (offset[:, 0] * self.stepsize[0]) ** 2 + \
                  (offset[:, 1] * self.stepsize[1]) ** 2
        gauss = np.exp(-0.5 * lengths / sigma**2)
        gauss = gauss / np.sum(gauss)
        offset[:, 0] = offset[:, 0] + 2 * int(sigma / self.stepsize[0])
        offset[:, 1] = offset[:, 1] + 2 * int(sigma / self.stepsize[1])
        for i in range(len(offset)):
            o = offset[i]
            outgrid[o[0]:o[0] + size[0], o[1]:o[1] + size[1]] += gauss[i] * grid
        l = []
        for i in range(len(outgrid)):
            for j in range(len(outgrid[0])):
                if outgrid[i, j] > 2000:
                    l.append([(i - zero2[0]) * self.stepsize[0], (j - zero2[1]) * self.stepsize[1], outgrid[i, j]])
        return l

    def save_random_error_machine(self, fp, sigma):
        rasterpoints = np.array(self.generate_random_error_machine(sigma))
        fp.write("submachine# %d %.2f %d %.1f\n" % (self.idx_energy, self.energy, self.idx_focus, self.focus))
        fp.write("#particles %.5E %.5E %.5E\n" % (np.min(rasterpoints[:, 2]), np.max(rasterpoints[:, 2]),
                                                  np.sum(rasterpoints[:, 2])))
        fp.write("stepsize %.0f %.0f\n" % (self.stepsize[0], self.stepsize[1]))
        fp.write("#points %d\n" % (len(rasterpoints)))
        for point in rasterpoints:
            fp.write("%d %d %.5E\n" % (point[0], point[1], point[2]))

    def load_submachine(self, data, i):
        items = data[i].split()
        self.idx_energy = int(items[1])
        self.energy = float(items[2])
        self.idx_focus = int(items[3])
        self.focus = float(items[4])

        while True:
            i += 1
            line = data[i]
            if line.find("stepsize") > -1:
                items = line.split()
                self.stepsize = [float(items[1]), float(items[2])]
            if line.find("#points") > -1:
                self.points = int(line.split()[1])
                break
            if line.find("#particles") > -1:
                self.min_particles = line.split()[1]
                self.max_particles = line.split()[2]
                self.total_particles = line.split()[3]
        i += 1
        for i in range(i, i + self.points):
            items = data[i].split()
            self.raster_points.append([float(items[0]), float(items[1]), float(items[2])])
        return i + 1
