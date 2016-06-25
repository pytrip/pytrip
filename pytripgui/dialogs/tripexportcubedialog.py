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
import pytrip
from wx.xrc import XmlResource, XRCCTRL, XRCID
import os, sys
import numpy as np

if getattr(sys, 'frozen', False):
    from wx.lib.pubsub import pub
    from wx.lib.pubsub import setuparg1
else:
    try:
        from wx.lib.pubsub import Publisher as pub
    except:
        from wx.lib.pubsub import setuparg1
        from wx.lib.pubsub import pub


class TripExportCubeDialog(wx.Dialog):
    def __init__(self):
        pre = wx.PreDialog()
        self.PostCreate(pre)

    def Init(self, plan):
        self.plan = plan
        self.path = "~/"
        self.output_path = ""
        self.checkbox_vois = XRCCTRL(self, "checkbox_vois")
        wx.EVT_LISTBOX(self, XRCID("checkbox_vois"), self.selected_changed)
        wx.EVT_BUTTON(self, XRCID("btn_ok"), self.save_and_close)
        wx.EVT_BUTTON(self, XRCID("btn_cancel"), self.close)
        wx.EVT_BUTTON(self, XRCID("btn_reset"), self.reset)

        self.lbl_path = XRCCTRL(self, "lbl_path")
        self.txt_value = XRCCTRL(self, "txt_value")
        wx.EVT_TEXT(self, XRCID("txt_value"), self.text_value_changed)

        pub.subscribe(self.path_changed, "general.export.cube_export_path")
        pub.sendMessage("settings.value.request", "general.export.cube")
        pub.subscribe(self.patient_data_updated, "patient.loaded")
        pub.sendMessage("patient.request", {})
        for voi in plan.get_vois():
            self.checkbox_vois.Append(voi.get_name())

    def patient_data_updated(self, msg):
        self.data = msg.data

    def selected_changed(self, evt):
        selected = self.checkbox_vois.GetStringSelection()
        self.txt_value.SetValue("")
        for voi in self.plan.get_vois():

            if selected == voi.get_name():
                if voi.get_cube_value() is -1:
                    self.txt_value.SetValue("")
                else:
                    self.txt_value.SetValue("%d" % voi.get_cube_value())

    def text_value_changed(self, evt):
        selected = self.checkbox_vois.GetStringSelection()
        if len(selected) is 0:
            return
        for voi in self.plan.get_vois():
            if selected == voi.get_name():
                try:
                    voi.set_cube_value(int(self.txt_value.GetValue()))
                except Exception as e:
                    pass

    def path_changed(self, msg):
        if not msg.data is None:
            self.path = msg.data
            self.lbl_path.SetLabel(self.path)

    def reset(self, evt):
        for voi in self.plan.get_vois():
            voi.set_cube_value(-1)
        self.txt_value.SetValue("")
        for k, item in enumerate(self.checkbox_vois.GetItems()):
            self.checkbox_vois.Check(k, False)

    def browse_for_file(self):
        dlg = wx.FileDialog(
            self,
            message="Save Picture",
            defaultDir=self.path,
            style=wx.FD_SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            a = os.path.splitext(path)
            if not a[-1] is "dos":
                path = path + ".dos"
                self.output_path = path
                pub.sendMessage("settings.value.updated", {"general.export.cube": os.path.dirname(path)})
            return True
        return False

    def save_and_close(self, evt):
        selected = self.checkbox_vois.GetCheckedStrings()
        vois = []
        dos = None
        for voi in self.plan.get_vois():
            if voi.get_name() in selected:
                if dos is None:
                    dos = voi.get_voi().get_voi_data().get_voi_cube() / 1000 * voi.get_cube_value()
                else:
                    dos.cube[dos.cube == 0] = -1;
                    a = voi.get_voi().get_voi_data().get_voi_cube() / 1000 * voi.get_cube_value()
                    dos.cube[dos.cube == -1] = a.cube[dos.cube == -1]

        if not dos is None:
            if self.browse_for_file():
                dos.write(self.output_path)
            else:
                return
        else:
            pass

        self.Close()

    def close(self, evt):
        self.Close()
