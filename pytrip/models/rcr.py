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
The RCR model is based on the paper from Antonovic et al.
https://doi.org/10.1093/jrr/rru020
Parameters are set for C-12 ions only.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def rcr_rbe(dose_ion, let, ax, bx, oxy=None):
    """
    Returns the RBE for a given dose/let cube.
    :params dose_ion: ion physical dose in [Gy]
    :params let: LET in [keV/um]
    :params ax: alpha for X-rays in [Gy^-1]
    :params bx: beta for X-rays in [Gy^-2]
    :params oxy: optional oxygenation cube in [mmHgO_2]
    """

    # Calculate sf_ion(D_ion, let, oxy)
    # from pytrip.models.aux import rbe_from_sfion
    # rbe_from_sfion(_sf, dose_ion, ax, bx)
    logger.warning("rcr_rbe not implemented yet.")
    pass  # TODO: not implemented yet.


def rcr_surviving_fraction(dose, let, oxy=None):
    """
    Function which returns surving fraction
    Equation (3) in https://doi.org/10.1093/jrr/rru020
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
        ocube = rcr_oer_po2(let, oxy)
        # ocube = rcr_oer(let)  # TODO: this was old version, not sure how to include.

    a = (a0 * _f(let) + a1 * np.exp(-let/ln)) / ocube
    b = b1 * np.exp(-let/ln) / ocube
    c = (c0 * _f(let) + c1 * np.exp(-let/ln)) / ocube

    sf = np.exp(-dose * a) + dose * b * np.exp(-dose * c)
    return sf


def _f(let):
    """
    f function from Dasu paper, takes let-cube as parameter
    Equation (7) in https://doi.org/10.1093/jrr/rru020
    """

    let[let == 0] = 0.01
    ld = 86.0
    return (1 - np.exp(-let/ld) * (1 + let/ld)) * ld/let


def rcr_oer(let):
    """
    ~O dose modifying factor.
    Equation (2) in https://doi.org/10.1093/jrr/rru020
    :returns: cube containing the oxygen enhancement ratio
    """

    omin = 1.10
    omax = 2.92
    lo = 114.0
    _o = omin + (omax - omin) * np.exp(-(let/lo)(let/lo))
    return _o


def rcr_oer_po2(let, oxy):
    """
    ~O dose modifying factor, taking varying pO2 into account
    Equation (1) in https://doi.org/10.1093/jrr/rru020
    :returns: cube containing the oxygen enhancement ratio
    """
    k = 2.5  # mmHg
    _o = rcr_oer(let) * (k + rcr_oer(let)) / (k + rcr_oer(let) * oxy)
    return _o
