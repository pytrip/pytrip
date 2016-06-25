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
from pytrip.vdx import *
from pytrip.dos import *
from pytrip.ctx import *
from pytrip.guiutil import *

c = CtxCube()
# c.read("/home/jato/Projects/TRiP/robustness/data/test1.ctx")
#
v = VdxCube(c)
# v.read_vdx("/home/jato/Projects/TRiP/robustness/data/test1.vdx")
#
# # ~ d.target_dose = 4.0
#g = PlotUtil()
# g.set_ct(c)
# # ~ add vois that should be plottet
# g.add_voi(v.get_voi_by_name("ptv"))
# # ~ g.add_voi(v.get_voi_by_name("tumor"))
# # ~ Plot slice number 80
# g.plot(80)
# g.plot(81)
pass