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
import sys

from pytrip.error import *
import os, sys

if getattr(sys, 'frozen', False):
    from wx.lib.pubsub import pub
    from wx.lib.pubsub import setuparg1
else:
    try:
        from wx.lib.pubsub import Publisher as pub
    except:
        from wx.lib.pubsub import setuparg1
        from wx.lib.pubsub import pub

from wx.xrc import XmlResource, XRCCTRL, XRCID


class TripExportDialog(wx.Dialog):
    def __init__(self):
        pre = wx.PreDialog()
        self.PostCreate(pre)

    def Init(self, plan):
        self.output_path = ''
        self.drop_type = XRCCTRL(self, "drop_type")
        self.txt_prefix = XRCCTRL(self, "txt_prefix")
        self.txt_prefix.SetValue(plan.get_name())

        self.label_folder = XRCCTRL(self, "label_folder")

        wx.EVT_BUTTON(self, XRCID("btn_cancel"), self.close)
        wx.EVT_BUTTON(self, XRCID("btn_browse"), self.browse_folder)
        wx.EVT_BUTTON(self, XRCID("btn_generate"), self.generate)
        self.plan = plan

        pub.subscribe(self.on_patient_updated, "patient.loaded")
        pub.subscribe(self.on_export_voxelplan_changed, "general.export.voxelplan")
        pub.sendMessage("settings.value.request", "general.export.voxelplan")
        pub.sendMessage("patient.request", None)

    def on_export_voxelplan_changed(self, msg):
        if not msg.data is None:
            self.output_path = msg.data
            self.label_folder.SetLabel(self.output_path)

    def on_patient_updated(self, msg):
        self.data = msg.data

    def browse_folder(self, evt):
        dlg = wx.DirDialog(
            self,
            defaultPath=self.output_path,
            message="Choose where the plan should be placed")
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.output_path = path
            pub.sendMessage("settings.value.updated", {"general.export.voxelplan": path})
            self.label_folder.SetLabel(path)

    def generate(self, evt):
        idx = self.drop_type.GetSelection()
        file_prefix = self.txt_prefix.GetValue()
        if len(file_prefix) == 0:
            raise InputError("File Prefix should be specified")
        if not hasattr(self, "output_path"):
            raise InputError("Output folder should be specified")
        path = os.path.join(self.output_path, file_prefix)
        exec_path = path + ".exec"
        ctx = self.data.get_images().get_voxelplan()
        if idx == 0:
            self.plan.save_data(ctx, path)
            self.plan.save_exec(ctx, exec_path)
        elif idx == 1:
            self.plan.save_exec(ctx, exec_path)
        elif idx == 2:
            self.plan.save_data(ctx, path)
        self.Close()

    def close(self, evt):
        self.Close()
