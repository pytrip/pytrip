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
TODO: documentation here.
"""
# do not check this file for PEP8 compatibility
# flake8 complains about "E402 module level import not at top of file"
# flake8: noqa

import logging
import os
import sys

from pytrip.cube import Cube
from pytrip.ctx import CtxCube
from pytrip.dos import DosCube
from pytrip.vdx import VdxCube, Voi
from pytrip.paths import DensityCube
from pytrip.raster import Rst
from pytrip.let import LETCube
from pytrip.dicomhelper import read_dicom_dir

# Python2 lacks a FileNotFoundError exception, we use then IOError
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

# from https://docs.python.org/3/tutorial/modules.html
# if a package's __init__.py code defines a list named __all__,
# it is taken to be the list of module names that should be imported when from package import * is encountered.
__all__ = ['CtxCube', 'VdxCube', 'Voi', 'DosCube', 'DensityCube', 'LETCube', 'dicomhelper', 'res',
           'Rst']

# if an application using pytrip doesn't configure any logging level, then an error will occur
# to prevent it, we add null logging handler, as suggested by Python documentation:
# as described here: https://docs.python.org/3/howto/logging.html#library-config
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = 'unknown'

# versioning of the package is based on __version__ variable, declared here in __init__.py file
# __version__ can be either calculated based on pytrip/VERSION file, storing version number
# or calculated directly from git repository (in case we work with cloned repo)

# we start by finding absolute path to the directory where pytrip98 is installed
# we cannot rely on relative paths as binary files which need version number may be located in different
# directory than the pytrip98 module

# get location of __init__.py file (the one you see now) in the filesystem
module_name = 'pytrip98'
if sys.version_info < (3, 0):  # Python 2.7
    import imp
    try:
        init_location = os.path.join(imp.find_module(module_name)[1], module_name)
    except ImportError:
        init_location = None
else:  # Python 3.x
    import importlib
    import importlib.util
    spec = importlib.util.find_spec(module_name)
    if spec:
        init_location = spec.origin
    else:
        init_location = None
if not init_location:
    init_location = os.path.dirname(os.path.realpath(__file__))


# VERSION should sit next to __init__.py in the directory structure
version_file = os.path.join(os.path.dirname(init_location), 'VERSION')

# let us check if pytrip98 was installed via pip and unpacked into some site-directory folder in the file system
# in such case we simply read the VERSION file
try:
    with open(version_file, 'r') as f:
        __version__ = f.readline().strip()  # read first line of the file (removing newline character)

# in case above methods do not work, let us assume we are working with GIT repository
except FileNotFoundError:
    # backup solution - read the version from git
    from pytrip.version import git_version
    __version__ = git_version()
