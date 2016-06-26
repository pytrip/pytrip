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
import wx
from wx.xrc import XmlResource, XRCCTRL, XRCID
from pytrip.vdx import *


class CreateStructureDialog(wx.Dialog):
    def __init__(self):
        pre = wx.PreDialog()
        self.PostCreate(pre)

    def Init(self, parent):
        self.parent = parent
        self.notebook = XRCCTRL(self, "notebook")
        voxelplan_image = self.parent.data.get_images().get_voxelplan()
        center = [voxelplan_image.dimx / 2.0 * voxelplan_image.pixel_size,
                  voxelplan_image.dimy / 2.0 * voxelplan_image.pixel_size,
                  voxelplan_image.dimz / 2.0 * voxelplan_image.slice_distance]

        self.txt_name = XRCCTRL(self, "txt_name")
        data = self.parent.data
        num = len(data.get_vois())
        num += 1
        self.txt_name.SetValue("ptv %d" % num)

        self.txt_x = XRCCTRL(self, "txt_x")
        self.txt_x.SetValue("%.1f" % center[0])

        self.txt_y = XRCCTRL(self, "txt_y")
        self.txt_y.SetValue("%.1f" % center[1])

        self.txt_z = XRCCTRL(self, "txt_z")
        self.txt_z.SetValue("%.1f" % center[2])

        # ini cube
        self.txt_width = XRCCTRL(self, "txt_width")
        self.txt_width.SetValue("50")

        self.txt_height = XRCCTRL(self, "txt_height")
        self.txt_height.SetValue("50")

        self.txt_depth = XRCCTRL(self, "txt_depth")
        self.txt_depth.SetValue("50")

        # ini cylinder
        self.txt_cylinder_radius = XRCCTRL(self, "txt_cylinder_radius")
        self.txt_cylinder_radius.SetValue("50")

        self.txt_cylinder_depth = XRCCTRL(self, "txt_cylinder_depth")
        self.txt_cylinder_depth.SetValue("40")

        # ini sphere
        self.txt_sphere_radius = XRCCTRL(self, "txt_sphere_radius")
        self.txt_sphere_radius.SetValue("25")

        wx.EVT_BUTTON(self, XRCID("btn_create"), self.submit)
        wx.EVT_BUTTON(self, XRCID("btn_cancel"), self.close)

    def close(self, evt):
        self.Close()

    def submit(self, evt):
        name = self.txt_name.GetValue()
        x = float(self.txt_x.GetValue())
        y = float(self.txt_y.GetValue())
        z = float(self.txt_z.GetValue())

        voi_type = self.notebook.GetCurrentPage().GetName()

        # cube
        width = float(self.txt_width.GetValue())
        height = float(self.txt_height.GetValue())
        depth = float(self.txt_depth.GetValue())

        # cylinder
        cylinder_radius = float(self.txt_cylinder_radius.GetValue())
        cylinder_depth = float(self.txt_cylinder_depth.GetValue())

        # sphere
        sphere_radius = float(self.txt_sphere_radius.GetValue())

        if voi_type == "panel_cube":
            voi = create_cube(self.parent.data.get_images().get_voxelplan(), name, [x, y, z], width, height, depth)
        elif voi_type == "panel_cylinder":
            voi = create_cylinder(self.parent.data.get_images().get_voxelplan(), name, [x, y, z], cylinder_radius,
                                  cylinder_depth)
        elif voi_type == "panel_sphere":
            voi = create_sphere(self.parent.data.get_images().get_voxelplan(), name, [x, y, z], sphere_radius)
        self.parent.data.load_voi(voi, True)
        self.Close()
