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

from pytrip.tripexecuter.dosecube import DoseCube
from pytrip.tripexecuter.field import Field
from pytrip.tripexecuter.fieldcollection import FieldCollection
from pytrip.tripexecuter.tripexecuter import TripExecuter
from pytrip.tripexecuter.tripplan import TripPlan
from pytrip.tripexecuter.tripplancollection import TripPlanCollection
from pytrip.tripexecuter.tripvoi import TripVoi
from pytrip.tripexecuter.voi import Voi
from pytrip.tripexecuter.voicollection import VoiCollection
from pytrip.tripexecuter.rbehandler import RBEHandler, RBE

# from https://docs.python.org/3/tutorial/modules.html
# if a package's __init__.py code defines a list named __all__,
# it is taken to be the list of module names that should be imported when from package import * is encountered.
__all__ = ['DoseCube', 'Field', 'FieldCollection', 'TripExecuter', 'TripPlan', 'TripPlanCollection', 'TripVoi', 'Voi',
           'VoiCollection', 'RBEHandler', 'RBE']
