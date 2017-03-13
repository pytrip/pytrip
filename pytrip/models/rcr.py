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
The RCR model is based on the paper from TODO.
"""

import numpy as np


def rcr_surviving_fraction(dose, let, oxy=None, model_parameters=None):
    """
    Function which returns surving fraction
    """
    
    if oxy is None:
        return _normoxic_survival(dose.cube, let.cube)

    else:
        return _hypoxic_survival(dose.cube, let.cube, oxy.cube)


def _f(let):
    """ f function from Dasu paper, takes let-cube as parameter
    """
    let[let == 0] = 0.01
    ld = 86
    return (1 - np.exp(-let/ld) * (1 + let/ld)) * ld/let


def rcr_oer(let):
    """ OER function from Dasu paper, takes let-cube as parameter
    :returns: cube containing the oxygen enhancement ratio
    """
    omin = 1.10
    omax = 2.92
    lo = 114
    _o = omin + (omax - omin) * np.exp(-(let/lo)(let/lo))
    return _o


def _normoxic_survival(dose, let):
    a0 = 5.7
    a1 = 1.3
    b1 = 2.0
    c0 = 5.7
    c1 = 0.2
    ln = 423

    a = a0 * _f(let) + a1 * np.exp(-let/ln)
    b = b1 * np.exp(-let/ln)
    c = c0 * _f(let) + c1 * np.exp(-let/ln)
    sv = np.exp(-dose * a) + dose * b * np.exp(-dose * c)
    return sv


def _hypoxic_survival(dose, let, oxy):
    """
    TODO: add support for heterogenous oxygeneation
    """
    a0 = 5.7
    a1 = 1.3
    b1 = 2.0
    c0 = 5.7
    c1 = 0.2
    ln = 423
    ocube = rcr_oer(let)

    a = (a0 * _f(let) + a1 * np.exp(-let/ln)) / ocube
    b = b1 * np.exp(-let/ln) / ocube
    c = (c0 * _f(let) + c1 * np.exp(-let/ln)) / ocube
    sv = np.exp(-dose * a) + dose * b * np.exp(-dose * c)
    return sv
