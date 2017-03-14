#
#    Copyright (C) 2010-2016 PyTRiP98 Developers.
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
Auxilliary tools
"""

import logging

logger = logging.getLogger(__name__)


def rbe_from_sf(sf_ion, dose_ion, ax, bx):
    """
    Returns the RBE for given ion survivng fraction and (alpha/beta)x-ray
    :params dose: ion physical dose in [Gy]
    :params sf_ion: surviving fraction in ion beam
    :params ax: alpha for X-rays in [Gy^-1]
    :params bx: beta for X-rays in [Gy^-2]
    """

    # Solve for D_x: bD2 + aD + ln(sf_ion) = 0
    # RBE = D_x / D_ion
    logger.warning("rcr_rbe not implemented yet.")
    pass  # TODO: not implemented yet.
