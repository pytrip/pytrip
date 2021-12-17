#
#    Copyright (C) 2010-2018 PyTRiP98 Developers.
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

import logging
import uuid

from pytrip.tripexecuter.kernel import KernelModel

logger = logging.getLogger(__name__)


class Field(object):
    """ One or more Field() object, which then can be added to a Plan() object.
    :params str basename: basename of field without file extension (input or output will be suffixed with
    proper file extension)
    """

    def __init__(self, kernel=KernelModel(), basename=""):
        """ Create an instance of the Field class.
        :params str basename" The name of this field, will be used as basename for .rst files.
        """
        self.__uuid__ = uuid.uuid4()  # for uniquely identifying this field
        # basename of the field (i.e. without file extension).
        # Any input/output process will be suffixed with .rst
        self.basename = basename
        self.number = 1  # Field number. First field must be 1
        self.gantry = 0.0  # TRiP98 angles assumed here.
        self.couch = 0.0  # TRiP98 angles assumed here.
        self.fwhm = 4.0  # spot width in [mm]
        self.raster_step = [2, 2]  # spot distance in [mm]
        self.dose_extension = 1.2  # see TRiP98 field / doseext() documentation
        self.contour_extension = 0.6  # see TRiP98 field / contourext() documentation

        # the field / command in TRiP98 may take an external rst file as an input, and will then
        # instead of optimization do a forward calculation based on this file.
        # The filename will be constructed from the basename given for this field.
        # However, we need to tell, that we provide the rst.
        self.use_raster_file = False

        # write out beams-eye-view file
        # if save_bev_file is True, then user can optionally provide a bev_filename
        # if filename is None, then it will be constructed from basename of the field
        self.save_bev_file = False
        self.bev_filename = None

        self.zsteps = 1.0  # in [mm]
        self.kernel = kernel

        # isocenter holds the [x,y,z] coordinates of the isocenter/target in [mm].
        # This is used for the field / target() option of the TRiP98
        # It can be used to override the automatically calculated isocenter from TRiP98.
        self.isocenter = []

    def __str__(self):
        """ str output handler
        """
        return self._print()

    def _print(self):
        """ Pretty print all attributes.
        """
        out = "\n"
        out += "   Field {:d} '{:s}'\n".format(self.number, self.basename)
        out += "----------------------------------------------------------------------------\n"
        out += "|  UUID                         : {:s}\n".format(str(self.__uuid__))
        out += "|  Couch angle                  : {:.2f} deg\n".format(self.couch)
        out += "|  Gantry angle                 : {:.2f} deg\n".format(self.gantry)
        out += "|\n"
        out += "|  Spot size (FWHM)             : {:.2f} mm\n".format(self.fwhm)
        out += "|  Raster step size (x,y)       : {:.2f}, {:.2f} mm\n".format(self.raster_step[0],
                                                                              self.raster_step[1])
        out += "|  Z-steps                      : {:.2f} mm\n".format(self.zsteps)
        out += "|\n"
        out += "|  Dose extension               : {:.2f}\n".format(self.dose_extension)
        out += "|  Contour extension            : {:.2f}\n".format(self.contour_extension)
        out += "|  Use external .rst file       : {:s}\n".format(str(self.use_raster_file))
        out += "|  Save .bev file               : {:s}\n".format(str(self.save_bev_file))
        if self.save_bev_file and self.bev_filename:
            out += "|  Beam eyes view filename      : {:s}\n".format(str(self.bev_filename))
        if self.isocenter:
            out += "|  Isocenter (x,y,z)            : {:.2f} {:.2f} {:.2f} mm\n".format(self.isocenter[0],
                                                                                        self.isocenter[1],
                                                                                        self.isocenter[2])
        else:
            out += "|  Isocenter (x,y,z)            : (not set)\n"
        out += "----------------------------------------------------------------------------\n"
        return out

    def set_isocenter_from_string(self, isocenter_str):
        """ Override the automatically determined isocenter from TRiP98.
        :params str isocenter_str: x,y,z coordinates of the isocenter/target in [mm] in a comma delimted string.
        such as "123.33,158.4,143.5".
        If and empty string is provided, then the isocenter is unset, and TRiP98 will calculate it itself.

        Following the target() option in the TRiP98 field command, one can specify
        the isocenter of the target. If not used, TRiP98 will determine the isocenter
        from the target Voi provided.

        """
        if len(isocenter_str) == 0:
            self.isocenter = []
            return
        _target = isocenter_str.split(",")
        if len(_target) == 3:
            try:
                self.isocenter = [float(_target[0]), float(_target[1]), float(_target[2])]
                return
            except ValueError:
                logger.error("Expected a 'X,Y,Z' formatted string for Field().set_isocenter_from_string")
