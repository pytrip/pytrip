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
TODO: documentation here.
"""
# do not check this file for PEP8 compatibility
# flake8 complains about "E402 module level import not at top of file"
# flake8: noqa

import logging
from pytrip.ctx import CtxCube
from pytrip.dos import DosCube
from pytrip.vdx import VdxCube, Voi
from pytrip.paths import DensityCube
from pytrip.raster import Rst
from pytrip.let import LETCube
from pytrip.ctimage import CTImages
from pytrip.dicomhelper import read_dicom_folder

# from https://docs.python.org/3/tutorial/modules.html
# if a package's __init__.py code defines a list named __all__,
# it is taken to be the list of module names that should be imported when from package import * is encountered.
__all__ = ['CtxCube', 'VdxCube', 'Voi', 'DosCube', 'DensityCube', 'LETCube', 'dicomhelper', 'res',
           'Rst', 'CTImages']

# if an application using pytrip doesn't configure any logging level, then an error will occur
# to prevent it, we add null logging handler, as suggested by Python documentation:
# as described here: https://docs.python.org/3/howto/logging.html#library-config
logging.getLogger(__name__).addHandler(logging.NullHandler())

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
