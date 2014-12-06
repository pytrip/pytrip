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
import sys
if getattr(sys, 'frozen', False):
    from wx.lib.pubsub import pub
else:
    try:
        from wx.lib.pubsub import Publisher as pub
    except:
        from wx.lib.pubsub import setuparg1
        from wx.lib.pubsub import pub

import pytrip.tripexecuter as pte
class DoseCube(pte.DoseCube):
    def __init__(self,dosecube,type):
        super(DoseCube,self).__init__(dosecube,type)
    def set_dose(self,value):
        super(DoseCube,self).set_dose(value)
        pub.sendMessage("plan.dose.target_dose_changed",self)

