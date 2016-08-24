#! /usr/bin/env python
"""Reads RST file from GSI
"""

import os
import re

# TODO add proper destructors


class SubMachine(object):
    """Define for each submachine."""

    def __init__(self, e_step, energy, focus_nr, focus_fl, subm_size):
        # print 'init submachine'
        self.energy_step = e_step
        self.energy = energy
        self.focus_step = focus_nr
        self.focus = focus_fl

        self.size = subm_size
        self.xpos = [0] * subm_size
        self.ypos = [0] * subm_size
        self.particles = [0] * subm_size

    def fillData(self, xx, yy, pp, n):
        # print "fill data:", n,  len(self.xpos)
        self.xpos[n] = xx
        self.ypos[n] = yy
        self.particles[n] = pp


class RstfileRead(object):
    """This class reads rst files."""

    FileIsRead = False

    def __init__(self, filename):
        """ Read the rst file."""
        # print 'initialized with filename',  filename

        if os.path.isfile(filename) is False:
            raise IOError("Could not find file " + filename)
        else:
            rstinput_file = open(filename, 'r')
            content = rstinput_file.readlines()
            # close(rstinput_file)
            data_length = len(content)
            print("read", data_length, "lines of data.")
            i = 0

            subm_counter = 0
            self.submachine = []
            while i < data_length:
                # print "parsing line",  i,  content[i]
                if re.match("sistable", content[i]) is not None:
                    self.sistable = content[i].split()[1]
                if re.match("rstfile", content[i]) is not None:
                    self.rstfile = content[i].split()[1]
                if re.match("patient_id", content[i]) is not None:
                    self.patient_id = content[i].split()[1]
                if re.match("projectile", content[i]) is not None:
                    self.projectile = content[i].split()[1]
                if re.match("charge", content[i]) is not None:
                    self.charge = int(content[i].split()[1])
                if re.match("mass", content[i]) is not None:
                    self.mass = int(content[i].split()[1])
                if re.match("gantryangle", content[i]) is not None:
                    self.gantryangle = float(content[i].split()[1])
                if re.match("couchangle", content[i]) is not None:
                    self.couchangle = float(content[i].split()[1])
                if re.match("stereotacticcoordinates", content[i]) is not None:
                    self.stereotactic = True
                else:
                    self.stereotactic = False
                if re.match("bolus", content[i]) is not None:
                    self.bolus = int(content[i].split()[1])
                if re.match("ripplefilter", content[i]) is not None:
                    self.ripplefilter = content[i].replace("ripplefilter ", "")
                if re.match("submachines", content[i]) is not None:
                    self.submachines = int(content[i].split()[1])

                if re.match("submachine#", content[i]) is not None:
                    # found a submachine, now do the data parsing
                    # print "found one submachine at",  i

                    # get an estimate of the submachine blocksize.
                    # It may be smaller, but certainly not larger than this.
                    i_stored = i
                    submachine_size = 0
                    i += 1
                    while (i < data_length) and (re.match("submachine#", content[i]) is None):
                        submachine_size += 1
                        i += 1
                    i = i_stored  # go back to the start of this submachine

                    data_counter = 0
                    self.submachine.append(
                        SubMachine(
                            int(content[i].split()[1]), float(content[i].split()[2]), int(content[i].split()[3]), float(
                                content[i].split()[4]), submachine_size))

                    i += 1
                    # now read the submachine
                    escape = False
                    while (i < data_length) and (re.match("submachine#", content[i]) is None) and (escape is False):
                        # print "subloop", i,  content[i]
                        if re.match("stepsize", content[i]) is not None:
                            self.submachine[subm_counter].stepsizex = int(content[i].split()[1])
                            self.submachine[subm_counter].stepsizey = int(content[i].split()[2])
                        else:
                            temp = content[i].split()
                            if len(temp) == 3:
                                try:
                                    x = float(temp[0])
                                    y = float(temp[1])
                                    z = float(temp[2])
                                    self.submachine[subm_counter].fillData(x, y, z, data_counter)
                                    data_counter += 1
                                except ValueError:
                                    # handles when some other comment is
                                    # inserted before the next "submachine"
                                    # line
                                    # print "escape", i, data_counter
                                    escape = True

                        i += 1
                    # number of entries in submachine
                    self.submachine[subm_counter].size = data_counter

                    # resize elements properly
                    # print "dc:", data_counter
                    self.submachine[subm_counter].xpos = self.submachine[subm_counter].xpos[:data_counter]

                    self.submachine[subm_counter].ypos = self.submachine[subm_counter].ypos[:data_counter]

                    self.submachine[subm_counter].particles = self.submachine[subm_counter].particles[:data_counter]
                    subm_counter += 1
                ################################
                else:
                    i += 1
            print("found", subm_counter, "submachines.")
            self.submachines = subm_counter

    def showVersion(self):
        print(self.version)


class SamfileRead(object):
    """This class reads rst files."""

    def __init__(self, filename):
        """ Read the rst file."""
        print('initialized with filename', filename)
        print("init")

    def __del__(self):
        object.__del__(self)
        print('deleted')

# example of usage:
# from rst_read import *
# a= RstfileRead("cub000101.rst")
# dir(a)
# print a.sistable
# print a.rstfile
# print a.couchangle
# print a.bolus
# print a.submachines
# b = a.submachine[0]
