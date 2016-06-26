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
from pytrip.ctx import *
from pytripgui.data import *


class CreateCubeDialog(wx.Dialog):
    def __init__(self):
        pre = wx.PreDialog()
        self.PostCreate(pre)

    def Init(self, parent):
        self.parent = parent
        self.txt_name = XRCCTRL(self, "txt_name")
        self.txt_name.SetValue("test")

        self.txt_hu = XRCCTRL(self, "txt_hu")
        self.txt_hu.SetValue("0")

        self.txt_dimx = XRCCTRL(self, "txt_dimx")
        self.txt_dimx.SetValue("512")

        self.txt_dimy = XRCCTRL(self, "txt_dimy")
        self.txt_dimy.SetValue("512")

        self.txt_dimz = XRCCTRL(self, "txt_dimz")
        self.txt_dimz.SetValue("100")

        self.txt_pixelsize = XRCCTRL(self, "txt_pixelsize")
        self.txt_pixelsize.SetValue("1")

        self.txt_slicedistance = XRCCTRL(self, "txt_slicedistance")
        self.txt_slicedistance.SetValue("3")

        wx.EVT_BUTTON(self, XRCID("btn_create"), self.submit)
        wx.EVT_BUTTON(self, XRCID("btn_cancel"), self.close)

    def close(self, evt):
        self.Close()

    def submit(self, evt):
        dimx = int(self.txt_dimx.GetValue())
        dimy = int(self.txt_dimy.GetValue())
        dimz = int(self.txt_dimz.GetValue())
        pixelsize = float(self.txt_pixelsize.GetValue())
        slice_distance = float(self.txt_slicedistance.GetValue())
        hu = int(self.txt_hu.GetValue())

        cube = CtxCube()
        cube.patient_name = self.txt_name.GetValue()
        cube.create_empty_cube(hu, dimx, dimy, dimz, pixelsize, slice_distance)
        self.parent.data = data.PytripData()
        self.parent.data.load_ctx_cube(cube)
        self.Close()
