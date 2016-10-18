"""
    This file is part of PyTRiP.

    PyTRiP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyTRiP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyTRiP.  If not, see <http://www.gnu.org/licenses/>
"""

# do not check this file for PEP8 compatibility
# flake8 complains about "E402 module level import not at top of file"
# flake8: noqa

import logging
from pytrip.ctx import CtxCube
from pytrip.dos import DosCube
from pytrip.vdx import VdxCube, Voi
from pytrip.paths import DensityCube
from pytrip.rst import Rst
from pytrip.let import LETCube
from pytrip.get_dvh import GetDvh
from pytrip.ctimage import CTImages

# from https://docs.python.org/3/tutorial/modules.html
# if a package's __init__.py code defines a list named __all__,
# it is taken to be the list of module names that should be imported when from package import * is encountered.
__all__ = ['CtxCube', 'VdxCube', 'Voi', 'GetDvh', 'DosCube', 'DensityCube', 'LETCube', 'dicomhelper', 'res',
           'Rst', 'CTImages']

# if an application using pytrip doesn't configure any logging level, then an error will occur
# to prevent it, we add null logging handler, as suggested by Python documentation:
# as described here: https://docs.python.org/3/howto/logging.html#library-config
logging.getLogger(__name__).addHandler(logging.NullHandler())