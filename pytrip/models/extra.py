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
Additional tools which are not tied to any particular model.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def rbe_from_sf(sf_ion, dose_ion, alpha_x, beta_x):
    """
    Returns the RBE for given ion survivng fraction and (alpha/beta)x-ray
    :params float dose: ion physical dose in [Gy] (cube or scalar)
    :params float sf_ion: surviving fraction in ion beam (cube of scalar)
    :params float alpha_x: alpha for X-rays in [Gy^-1] (cube or scalar)
    :params float beta_x: beta for X-rays in [Gy^-2] (cube or scalar)
    """

    # Solve for D_x: beta_x*D*2 + alpha_x*D + ln(sf_ion) = 0
    # RBE = D_x / D_ion

    a = beta_x * dose_ion * 2.0
    b = alpha_x * dose_ion
    c = np.log(sf_ion)  # natural logarithm
    d = b * b - 4 * a * c
    x1 = (-b + np.sqrt(d)) / (2.0 * a)
    x2 = (-b - np.sqrt(d)) / (2.0 * a)

    if x1 > x2:
        dose_x = x1
    else:
        dose_x = x2

    rbe = dose_x / dose_ion

    if rbe < 0:
        logger.warning("negative rbe encountered")
    return rbe


def lq(dose_x, alpha_x, beta_x):
    """
    Linear-quadratic survival model. (LQ-model)

    Returns surviving fraction for a given dose, alpha and beta for X-rays.

    :params float dose: x-ray physical dose in [Gy] (cube or scalar)
    :params float alpha_x: alpha value for x-rays [Gy^-1] (cube or scalar)
    :params float beta_x: beta value for x-rays [Gy^-2] (cube or scalar)
    """
    return np.exp(-alpha_x * dose_x - beta_x * dose_x * dose_x)
