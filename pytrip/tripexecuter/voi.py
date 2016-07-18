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
import copy


class Voi(object):
    def __init__(self, name, voi):
        self.name = name
        self.selected = False
        self.voxelplan_voi = voi

    def get_color(self):
        return self.voxelplan_voi.get_color()

    def is_selected(self):
        return self.selected

    def toogle_selected(self):
        self.selected = not self.selected

    def get_voi_data(self):
        return self.voxelplan_voi

    def get_name(self):
        return self.name

    def copy(self):
        return copy.deepcopy(self)
