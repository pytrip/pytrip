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
The RCR model is based on the paper from Antonovic et al.
https://doi.org/10.1093/jrr/rru020
Parameters are set for C-12 ions only.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def rbe_rcr(dose_ion, let, alpha_x, beta_x, oxy=None):
    """
    Returns the RBE for a given dose/let cube.

    input parameters may be either numpy.array or scalars
    TODO: handle real cubes.

    :params dose_ion: ion physical dose in [Gy]
    :params let: LET in [keV/um]
    :params alpha_x: alpha for X-rays in [Gy^-1]
    :params beta_x: beta for X-rays in [Gy^-2]
    :params oxy: optional oxygenation cube in [mmHgO_2]
    """

    # Calculate sf_ion(D_ion, let, oxy)
    # from pytrip.models.extra import rbe_from_sf
    # rbe_from_sf(_sf, dose_ion, alpha_x, beta_x)
    logger.warning("rcr_rbe not implemented yet.")


def sf_rcr(dose, let, oxy=None):
    """
    Function which returns surving fraction
    Equation (3) in https://doi.org/10.1093/jrr/rru020

    input parameters may be either numpy.array or scalars
    TODO: handle real cubes.

    :params dose: physical ion dose in [Gy]
    :params let: LET in keV/um
    :params oxy: optional oxygenation in [mmHgO_2]
    """

    a0 = 5.7
    a1 = 1.3
    b1 = 2.0
    c0 = 5.7
    c1 = 0.2
    ln = 423.0

    if oxy is None:
        ocube = 1.0
    else:
        ocube = oer_po2_rcr(let, oxy)
        # ocube = oer_rcr(let)  # TODO: this was old version, not sure how to include.

    a = (a0 * _f(let) + a1 * np.exp(-let / ln)) / ocube
    b = b1 * np.exp(-let / ln) / ocube
    c = (c0 * _f(let) + c1 * np.exp(-let / ln)) / ocube

    sf = np.exp(-dose * a) + dose * b * np.exp(-dose * c)
    return sf


def _f(let):
    """
    f function from Dasu paper, takes let-cube as parameter
    Equation (7) in https://doi.org/10.1093/jrr/rru020

    input parameters may be either numpy.array or scalars
    TODO: handle real cubes.

    :params let: LET in [keV/um]

    :returns: result of the f function
    """

    ld = 86.0
    result = (1 - np.exp(-let / ld) * (1 + let / ld)) * ld / let

    # map any zero LET areas to 0.0
    if np.isscalar(result):  # scalar
        if result == np.inf:
            result = 0.0
    else:
        result[result == np.inf] = 0.0  # numpy arrays

    return result


def oer_rcr(let):
    """
    ~O dose modifying factor.
    Equation (2) in https://doi.org/10.1093/jrr/rru020

    input parameters may be either numpy.array or scalars
    TODO: handle real cubes.

    :params let: LET in [keV/um]

    :returns: cube containing the oxygen enhancement ratio
    """

    omin = 1.10
    omax = 2.92
    lo = 114.0
    _o = omin + (omax - omin) * np.exp(-(let / lo) * (let / lo))
    return _o


def oer_po2_rcr(let, oxy):
    """
    ~O dose modifying factor, taking varying pO2 into account
    Equation (1) in https://doi.org/10.1093/jrr/rru020

    input parameters may be either numpy.array or scalars
    TODO: handle real cubes.

    :params let: LET in [keV/um]
    :params oxy: oxygenation in [mmHgO_2]

    :returns: cube containing the oxygen enhancement ratio
    """
    k = 2.5  # mmHg
    _o = oer_rcr(let) * (k + oer_rcr(let)) / (k + oer_rcr(let) * oxy)
    return _o
