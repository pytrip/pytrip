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
import pytrip as pt

# read a dose cube, divide by 2.0, and write to a new cube:
d0 = pt.DosCube()
d0.read("box050001.dos")
d0 = d0 / 2.0
d0.write("out0.dos")

# sum two dose cubes, write result:
print("Two half boxes: out.dos")
d1 = pt.DosCube()
d2 = pt.DosCube()
d1.read("box052000.dos")
d2.read("box053000.dos")
d = (d1 + d2)
d.write("out.dos")

# print minium and maximum value found in cubes
print(d1.cube.min(), d1.cube.max())
print(d0.cube.min(), d0.cube.max())

# calculate new dose average LET cube
l1 = pt.LETCube()
l2 = pt.LETCube()
l1.read("box052000.dosemlet.dos")
l2.read("box053000.dosemlet.dos")

l = ((d1 * l1) + (d2 * l2)) / (d1 + d2)
l.write("out.dosemlet.dos")

# load a vdx
v = pt.VdxCube("", d0)
v.read("contours.vdx")
