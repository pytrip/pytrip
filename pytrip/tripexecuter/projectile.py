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
Object holding all data needed for a specific projectile/rifi configuration.
"""

import logging


logger = logging.getLogger(__name__)


class Projectile(object):
    """
    Object holding projectile specific data.
    """

    # list of projectile name - charge and most common isotope.
    _projectile_defaults = {"H": (1, 1),
                            "He": (2, 4),
                            "Li": (3, 7),
                            "C": (6, 12),
                            "O": (8, 16),
                            "Ne": (10, 20),
                            "Ar": (16, 40)}

    z = -1
    a = -1
    iupac = ""  # chemical symbol (parsed by TRiP), "H", "He", "C", "O", "Ne" ....
    name = ""   # cleartext name "Protons", "Carbon-12 ions", "Antiprotons", ...

    def __init__(self, symb, name="", z=0, a=0):
        """
        :param symb: IUPAC symbol "H", "He", "Li", "C" ...
        :param name: free text name for this projectile, i.e. "Protons", "Antiprotons", "C-12"
        :param z: charge of projectile. Default value for symb taken if not specified.
        :param a: nucleon number of proejctile, default value for symb is taken if not specified (12 for C, 4 for He...)
        """

        self.name = name

        if symb in self._projectile_defaults:
            self.z = self._projectile_defaults[symb][0]
            self.a = self._projectile_defaults[symb][1]

        self.iupac = symb

        if z > 0:
            self.z = z

        if a > 0:
            self.a = a

        if self.z < 1:
            logger.warning("No projectile charge was set.")
        if self.a < 1:
            logger.warning("No projectile nucleon number was set.")

    def __str__(self):
        """
        String out handler
        """
        return self._print()

    def _print(self):
        """
        Pretty print all attributes.
        """
        out = "\n"
        out += "   Projectile '{:s}'\n".format(self.name)
        out += "----------------------------------------------------------------------------\n"
        out += "|  Projectile Z                      : {:s}\n".format(str(self.z))
        out += "|  Projectile A                      : {:s}\n".format(str(self.a))
        out += "|  Projectile IUPAC                  : {:s}\n".format(str(self.iupac))
        out += "----------------------------------------------------------------------------\n"
        return out
