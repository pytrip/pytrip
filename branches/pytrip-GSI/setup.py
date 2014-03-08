
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
from distutils.core import setup,Extension
import numpy as np
requires = [
    'matplotlib>=0.99',
    'numpy>=1.2.1',
    'pydicom>=0.9.5',]



module1 = Extension('pytriplib',
                    sources = ['pytrip/lib/filter_point.c'])

setup (name = 'pytriplib',
       version = '0.1',
       include_dirs = [np.get_include()],
       description = 'help functions for pytrip',
       ext_modules = [module1])

setup(name='pytrip',
      description='Python scripts for TRiP and virtuos',
      author='Jakob Toftegaard, Niels Bassler',
      author_email='bassler@phys.au.dk',
      url='http://aptg-trac.phys.au.dk/pytrip/',
      # TODO: this does not look correct, 
      # why do the subdirectories have to be specified explicitly here?
      packages=['pytrip','pytrip.res','pytrip.tripexecuter',
                'pytripgui','pytripgui.dialogs','pytripgui.panels'], 
      #
      #install_requires = requires,
      package_data={'pytrip': ['data/*.dat'], 
                    'pytripgui' : ['res/*']},
      scripts=['scripts/pytrip-gui'
               ],
      version='0.1',
      )
