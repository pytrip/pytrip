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
Structure:
RBEBaseData() holds a path and base name of the .rbe file, and parses the set.
"""

import os
import logging

from pytrip.error import FileNotFound

logger = logging.getLogger(__name__)


class RBEBaseData(object):
    """
    Class for TRiP .rbe files.
    """

    def __init__(self, basename=""):
        """
        """
        self.path = ""
        self.basename = ""  # basename derived from path without file extension or leading directory
        self.fileversion = ""
        self.filedate = ""
        self.celltype = ""
        self.mean = False
        self.alpha = 0.0  # Photon alpha [Gy^-2]
        self.beta = 0.0  # Photon beta [Gy^-1]
        self.cut = 0.0  # Cutoff energy, above which where cell response changes to linear behaviour [Gy]
        self.rnucleus = 0.0  # radius of nucleus in um
        self.opposingfieldcoeff = (0.0, 0.0)

    def __str__(self):
        """ str output handler
        """
        return self._print()

    def _print(self):
        """
        Pretty print all attributes.
        """
        out = "\n"
        out += "   RBEBaseData '{:s}'\n".format(self.celltype)
        out += "----------------------------------------------------------------------------\n"
        out += "|  Path                         : '{:s}'\n".format(str(self.path))
        out += "|  Basename                     : '{:s}'\n".format(str(self.basename))
        out += "|  File version                 : '{:s}'\n".format(str(self.fileversion))
        out += "|  File date                    : '{:s}'\n".format(str(self.filedate))
        out += "|  Cell type                    : '{:s}'\n".format(str(self.celltype))
        out += "|  Mean                         : {:s}\n".format(str(self.mean))
        out += "|\n"
        out += "|  alpha_x                      : {:.3f} [Gy-2]\n".format(self.alpha)
        out += "|  beta_x                       : {:.3f} [Gy-1]\n".format(self.beta)
        if (self.beta) != 0.0:
            out += "|  (a/b)_x                      : {:.2f}  [Gy]\n".format(self.alpha / self.beta)
        out += "|  Dose cutoff                  : {:.2f} [Gy]\n".format(self.cut)
        out += "|  Nucleus radius               : {:.2f} [um]\n".format(self.rnucleus)
        out += "|  Opposing field coeff.        : {:.2f} {:.2f}\n".format(self.opposingfieldcoeff[0],
                                                                          self.opposingfieldcoeff[1])
        out += "----------------------------------------------------------------------------\n"
        return out

    def read(self, path):
        """
        :param path: string pointing to filename
        """
        if not os.path.exists(path):
            raise FileNotFound("Loading {:s} failed, file not found".format(path))
        self._read_trip_rbe_file(path)

    def _read_trip_rbe_file(self, path):
        """
        Read and parse .rbe file given by path. Populate attributes.
        """
        logger.debug("Opening {:s}".format(path))

        self.path = path
        self.basename = os.path.splitext(os.path.split(path)[1])[0]

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # there may be several whitespace in the payload, so strip() wont work.
            # Instead, the token is identified.
            items = line.split()
            if len(items) > 1:
                token = line.split()[0]
                payload = ""
                if len(token) + 1 < len(line):
                    payload = line[len(token) + 1:].strip()

                if "!filetype" in token:
                    self.filetype = payload
                if "!fileversion" in token:
                    self.fileversion = payload
                if "!filedate" in token:
                    self.filedate = payload
                if "!celltype" in token:
                    self.celltype = payload
                if "!mean" in token:
                    self.mean = True
                if "!alpha" in token:
                    self.alpha = float(payload)
                if "!beta" in token:
                    self.beta = float(payload)
                if "!cut" in token:
                    self.cut = float(payload)
                if "!rnucleus" in token:
                    self.rnucleus = float(payload)
                if "!opposingfieldcoeff" in token:
                    _of = payload.split()
                    self.opposingfieldcoeff = (float(_of[0]), float(_of[1]))

            def _parse_projectiles(self, lines):
                """
                TODO: not implemented
                """
                pass

            def _parse_sumdoseptable(self, lines):
                """
                TODO: not implemented
                """
                pass
