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
    """ This class handles raster scan data, which are accelerator control files in GSI format.
    Raster scan data are stored in .rst file, and describe the amount of particles going into
    each spot in each energy layer. Each energy layer is called a 'submachine'.
    """
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
        """
        :returns: A list of submachines.
        """
        return self.machines

    def calculate_total_energy(self):  # TODO: not implemented
        """
        Not implemented.
        """
        return

    def load_dicom(self, path):  # TODO: not implemented, load_dicom() -> read_from_dicom()
        """ Load a Dicom file from 'path'

        Currently, this function merely stores the dicom data into self.data.
        No interpretation is done.

        :param str path: Full path to Dicom file.
        """
        self.data = dicom.read_file(path)

    def load_field(self, path):  # TODO: load_field() -> read()
        """ Load and parse a raster scan (.rst) file.

        :param str path: Full path to the file to be loaded, including file extension.
        """
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
            i = submachine._load_submachine(data, i)
            self.machines.append(submachine)

    def get_stepsize(self):
        """ Returns the distance between each spot in the first energy plane.

        Most likely the distance will be the same in all planes.

        :returns: Distancce between spots in [mm]. If no submachines are found, None is returned.
        """
        if len(self.machines) > 0:
            return self.machines[0].stepsize
        return None

    def get_min_max(self):
        """ Retrieve the largest and smallest x,y position found in all energy layers.

        :returns: A list of four values in [x_min,x_max,y_min,y_max] in [mm].
        """
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
        """ Returns the smallest and largest x and y positions for this energy layer.
        :returns: a list of four elements [min_x, max_x, min_y, max_y]
        """
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
        """ Generates a new set of raster points, including distributed setup errors with a Gaussian sigma.

        This function is used to simulate positioning errors.
        The existing raster field is shifted using a Gaussian distribution. This corresponds to blurring the
        raster scan file.
        Spot weights below 2000 particles are truncated.

        :param float sigma: sigma of the Gaussian blur to be applied [mm]
        """
        # prepare a grid which covers (xmin,ymin) to (xmax,ymax)
        min_max = self.get_raster_min_max()
        zero = [-int(min_max[0] / self.stepsize[0]),
                -int(min_max[2] / self.stepsize[1])]  # start position of steps [i_min,j_min], may be negative
        size = [(min_max[1] - min_max[0]) / self.stepsize[0] + 1,
                (min_max[3] - min_max[2]) / self.stepsize[1] + 1]  # no of steps to reach opposite side [i_len,j_len]
        grid = np.zeros((size))  # 2d grid with steop positions

        # Fill the new grid with raster values from the original raster points
        rasterpoints = np.array(self.raster_points)  # cast raster points into numpy array
        grid[np.array(rasterpoints[:, 0] / self.stepsize[0] + zero[0], 'uint8'),
             np.array(rasterpoints[:, 1] / self.stepsize[1] + zero[1], 'uint8')] = rasterpoints[:, 2]

        # extend grid size a bit to cover positions going beyond the edges when applying an offset.
        # determine the new size:
        size2 = [size[0] + 4 * int(sigma / self.stepsize[0]),  # width
                 size[1] + 4 * int(sigma / self.stepsize[1])]
        zero2 = [zero[0] + 2 * int(sigma / self.stepsize[0]),  # start positions
                 zero[1] + 2 * int(sigma / self.stepsize[1])]

        # make new grid with new size
        outgrid = np.zeros(size2)

        # calculate a few offset positions from a Gaussian.
        # offset holds offset indices. Mostly this will be just [-1,0,1] for x and y.
        offset = np.array([[x, y]
                           for x in range(-int(sigma / self.stepsize[0]), int(sigma / self.stepsize[0]) + 1)
                           for y in range(-int(sigma / self.stepsize[1]), int(sigma / self.stepsize[1]) + 1)])

        # Translate integer offsets to actual positions, squared.
        lengths = (offset[:, 0] * self.stepsize[0]) ** 2 + \
                  (offset[:, 1] * self.stepsize[1]) ** 2

        gauss = np.exp(-0.5 * lengths / sigma ** 2)  # Array shaping a Gaussian distribution from lengths
        gauss = gauss / np.sum(gauss)  # Normalize it

        # extend number of offsets
        offset[:, 0] = offset[:, 0] + 2 * int(sigma / self.stepsize[0])  # x
        offset[:, 1] = offset[:, 1] + 2 * int(sigma / self.stepsize[1])  # y

        # For each position, modify the spot intensity to be a sum of the offsets.
        for i in range(len(offset)):
            o = offset[i]
            outgrid[o[0]:o[0] + size[0], o[1]:o[1] + size[1]] += gauss[i] * grid

        # store new grid in l which will be returned
        l = []
        for i in range(len(outgrid)):
            for j in range(len(outgrid[0])):  # run along y indices
                # only append to result grid l, if we have a reasonable number of particles in this spot.
                if outgrid[i, j] > 2000:  # spot threshold is 2000 particles
                    l.append([(i - zero2[0]) * self.stepsize[0],
                              (j - zero2[1]) * self.stepsize[1],
                              outgrid[i, j]])
        return l

    def save_random_error_machine(self, fp, sigma):
        """ Generates and stores a single energy layer where Gaussian blur has been applied.

        :param fp: file pointer
        :param float sigma: sigma of the Gaussian blur to be applied [mm]
        """
        rasterpoints = np.array(self.generate_random_error_machine(sigma))
        fp.write("submachine# %d %.2f %d %.1f\n" % (self.idx_energy, self.energy, self.idx_focus, self.focus))
        fp.write("#particles %.5E %.5E %.5E\n" % (np.min(rasterpoints[:, 2]), np.max(rasterpoints[:, 2]),
                                                  np.sum(rasterpoints[:, 2])))
        fp.write("stepsize %.0f %.0f\n" % (self.stepsize[0], self.stepsize[1]))
        fp.write("#points %d\n" % (len(rasterpoints)))
        for point in rasterpoints:
            fp.write("%d %d %.5E\n" % (point[0], point[1], point[2]))

    def _load_submachine(self, data, i):
        """ Loads a single submachine from 'data'.

        :param [str] data: list of lines containing the .rst data.
        :param i: line number to start from
        :returns: new line number, when parsing has finished.
        """
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
