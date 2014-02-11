"""
    This file is part of PyTRiP.

    libdedx is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    libdedx is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with libdedx.  If not, see <http://www.gnu.org/licenses/>
"""
from distutils.core import setup, Extension
import numpy as np

module1 = Extension('pytriplib',
                    sources = ['filter_point.c'],
                    extra_compile_args=['-fopenmp', '-fpic'],
                    extra_link_args=['-lgomp'])
setup (name = 'pytriplib',
       version = '0.1',
       include_dirs = [np.get_include()],
       description = 'help functions for pytrip',
       ext_modules = [module1])
