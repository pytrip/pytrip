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
TCP models
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def tcp_voi(sf, voi=None, ncells=1.0, fractions=1):
    """
    Returns TCP within VOI.
    If VOI is not give, TCP of entire cube is calculated.
    This is equation (7) in https://doi.org/10.1093/jrr/rru020
    assuming static oxygenation during all fractions.
    (Equation (8) would require a new oxy cube after every fractionation, not implemented.)

    :params numpy.array sf: numpy array, surviving fraction cube
    :params Voi voi: pytrip Voi() class object
    :params float ncells: number of cells in each voxel, or a cube of surviving fractions
    :params int fractions: number of fractions, default is 1
    """

    tcp = 0

    if voi is None:
        mask = None
    else:
        # mark ony those values as true, which are within the VOI
        voi_cube = voi.get_voi_cube()
        mask = (voi_cube.cube == 1000)

    # ncells may be either a scalar or a cube.
    if np.isscalar(ncells):
        tcp = np.exp(-sum(ncells * sf[mask]**fractions))
    else:
        # better make sure that the cells cube has the same size as the surviving fraction cube
        if ncells.shape == sf.shape:
            tcp = np.exp(-sum(ncells[mask] * sf[mask]**fractions))
        else:
            logger.error("ncells array shape does not match surviving fraction shape.")

    return tcp
