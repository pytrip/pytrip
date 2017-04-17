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


class Field():
    """ One or more Field() object, which then can be added to a Plan() object.
    :params str basename: basename of field without file extension (input or output will be suffixed with
    proper file extension)
    """
    def __init__(self, basename=""):
        self.__uuid__ = uuid.uuid4()  # for uniquely identifying this field
        # basename of the field (i.e. without file extension).
        # Any input/output process will be suffixed with .rst
        self.basename = basename
        self.gantry = 0.0  # TRiP98 angles assumed here.
        self.couch = 0.0  # TRiP98 angles assumed here.
        self.fwhm = 4.0  # in [mm]
        self.rasterstep = [2, 2]
        self.doseextension = 1.2
        self.contourextension = 0.6

        self.zsteps = 1.0  # in [mm]
        self.projectile = 'C'  # see also self._projectile_defaults
        self.projectile_a = '12'  # Number of nucleons in projectile. If None, default will be used.

        # isocenter holds the [x,y,z] coordinates of the isocenter/target in [mm].
        # This is used for the field / target() option of the TRiP98
        # It can be used to override the automatically calculated isocenter from TRiP98.
        self.isocenter = []

        self.selected = False

        # list of projectile name - charge and most common isotope.
        self._projectile_defaults = {"H": (1, 1),
                                     "He": (2, 4),
                                     "Li": (3, 7),
                                     "C": (6, 12),
                                     "O": (8, 16),
                                     "Ne": (10, 20),
                                     "Ar": (16, 40)}

    def set_isocenter_from_string(self, isocenter_str):
        """ Override the automatically determined isocenter from TRiP98.
        :params str isocenter_str: x,y,z coordinates of the isocenter/target in [mm] in a comma delimted string.
        such as "123.33,158.4,143.5".
        If and empty string is provided, then the isocenter is unset, and TRiP98 will calculate it itself.

        Following the target() option in the TRiP98 field command, one can specify
        the isocenter of the target. If not used, TRiP98 will determine the isocenter
        from the target Voi provided.

        """
        if len(isocenter_str) is 0:
            self.isocenter = []
            return
        _target = isocenter_str.split(",")
        if len(_target) is 3:
            try:
                self.isocenter = [float(_target[0]), float(_target[1]), float(_target[2])]
                return
            except Exception:
                logger.error("Expected a 'X,Y,Z' formatted string for Field().set_isocenter_from_string")
