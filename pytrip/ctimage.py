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
This module provides the CTImages class.
"""
import copy


class CTImages:
    def __init__(self, voxelplan):
        self.voxelplan_images = voxelplan

    def get_voxelplan(self):
        return self.voxelplan_images

    def get_modified_images(self, plan):
        modify = False
        for voi in plan.get_vois():
            if not voi.get_hu_offset() is None or not voi.get_hu_value() is None:
                modify = True
                break
        if modify:
            images = copy.deepcopy(self.voxelplan_images)
            for voi in plan.get_vois():
                if not voi.get_hu_offset() is None:
                    images.set_offset_cube_values(voi.get_voi().get_voi_data(), voi.get_hu_offset())
                if not voi.get_hu_value() is None:
                    images.override_cube_values(voi.get_voi().get_voi_data(), voi.get_hu_value())
        else:
            images = self.voxelplan_images
        return images
