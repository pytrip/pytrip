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
Collection of proton RBE models.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def rbe_carabe(dose, let, abx):
    """
    Carabe proton RBE model
    :ref:
    """

    _labx = 2.686 * let / abx
    _apx = 0.843 + 0.154 * _labx
    _bpx = 1.090 + 0.006 * _labx
    _bpx *= _bpx

    rbe = _rbe_apx(dose, _apx, _bpx, abx)
    return rbe


def rbe_wedenberg(dose, let, abx):
    """
    Wedenberg proton RBE model
    :ref:
    """

    _apx = 1.000 + 0.434 * let / abx
    _bpx = 1.000

    rbe = _rbe_apx(dose, _apx, _bpx, abx)
    return rbe


def rbe_mcnamara(dose, let, abx):
    """
    McNamara proton RBE model
    :ref:
    """

    _apx = 0.999064 + 0.35605 * let / abx
    _bpx = 1.1012 - 0.0038703 * np.sqrt(abx) * let
    _bpx *= _bpx

    rbe = _rbe_apx(dose, _apx, _bpx, abx)
    return rbe


def _rbe_apx(dose, apx, bpx, abx):
    """
    :params dose: proton dose
    :params apx: alpha_p / alpha_x
    :params bpx: beta_p / beta_x
    :params abx: alpha_x / beta_x
    """

    rbe = np.sqrt(abx*abx + 4*apx*abx*dose + 4*bpx*dose*dose - abx) / (2 * dose)
    return rbe
