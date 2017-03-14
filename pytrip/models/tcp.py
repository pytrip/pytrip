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
TCP models
"""

import logging

logger = logging.getLogger(__name__)


def tcp_voi(sf, voi=None, ncells=1.0, fractions=1):
    """
    Returns TCP within VOI.
    If VOI is not give, TCP of entire cube is calculated.
    This is equation (7) in https://doi.org/10.1093/jrr/rru020
    assuming static oxygenation during all fractions.
    (Equation (8) would require a new oxy cube after every fractionation, not implemented.)
    """

    # _sf = rcr_surviving_fraction(dose, let, oxy)

    # TODO: extract masked array
    # _tcp = np.exp( -sum(ncells * (_sf)^fractions))
    logger.warning("tcp_voi() not implemented yet")
    pass
