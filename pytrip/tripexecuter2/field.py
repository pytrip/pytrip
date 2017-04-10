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
Field() objects here, which can be passed to a Plan() object.
"""

import uuid
import logging

logger = logging.getLogger(__name__)


class Field(pytripObj):
    """ One or more Field() object, which then can be added to a Plan() object.
    """
    def __init__(self, name=""):
        self.__uuid__ = uuid.uuid4()  # for uniquely identifying this field
        self.name = name
        self.gantry = 0.0  # TRiP98 angles assumed here.
        self.couch = 0.0  # TRiP98 angles assumed here.
        self.fwhm = 4.0  # in [mm]
        self.rasterstep = [2, 2]
        self.doseextension = 1.2
        self.contourextension = 0.6
        self.rasterfile_path = None

        self.zsteps = 1.0  # in [mm]
        self.projectile = 'C'
        self.projectile_a = None  # Number of nucleons in projectile. If None, default will be used.
        self.target = []
        self.selected = False

        # list of projectile name - charge and most common isotope.
        self._projectile_defaults= {"H": (1,1),
                                    "He": (2,4),
                                    "Li": (3,7),
                                    "C": (6,12),
                                    "O": (8,16),
                                    "Ne": (10,20)
                                    "Ar": (16,40)}

    def set_isocenter(self, target):
        """ Override the automatically determined isocenter from TRiP98.
        :params [float, float, float] target: [x,y,z] coordinated of the isocenter/target in [mm]
        """
        self.target = target
        
    def set_isocenter_from_string(self, target):
        """ Override the automatically determined isocenter from TRiP98.
        :params str target: x,y,z coordinates of the isocenter/target in [mm] in a comma delimted string.

        Following the target() option in the TRiP98 field command, one can specify
        the isocenter of the target. If not used, TRiP98 will determine the isocenter 
        from the target Voi provided.
        This function is similar to Field.set_isocenter(), but takes a string as input argument.
        """
        if len(target) is 0:
            self.target = []
            return
        target = target.split(",")
        if len(target) is 3:
            try:
                self.target = [float(target[0]), float(target[1]), float(target[2])]
                return
            except Exception:
                logger.error("Expected a 'X,Y,Z' formatted string for Field().target")

        raise InputError("Target should be empty " "or in the format x,y,z")
