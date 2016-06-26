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

if getattr(sys, 'frozen', False):
    from wx.lib.pubsub import pub
else:
    try:
        from wx.lib.pubsub import Publisher as pub
    except:
        from wx.lib.pubsub import setuparg1
        from wx.lib.pubsub import pub

from wx.xrc import XmlResource, XRCCTRL, XRCID


class TripVoiDialog(wx.Dialog):
    def __init__(self):
        pre = wx.PreDialog()
        self.PostCreate(pre)
        pub.subscribe(self.patient_data_updated, "patient.loaded")
        pub.sendMessage("patient.request", {})

    def patient_data_updated(self, msg):
        self.data = msg.data

    def select_drop_by_value(self, drop, value):
        for i, item in enumerate(drop.GetItems()):
            if item == value:
                drop.SetSelection(i)

    def Init(self, voi):
        self.voi = voi

        wx.EVT_BUTTON(self, XRCID('btn_ok'), self.save_and_close)
        wx.EVT_BUTTON(self, XRCID('btn_close'), self.close)

        self.label_name = XRCCTRL(self, "label_name")
        self.label_name.SetLabel(voi.get_name())

        self.txt_dose = XRCCTRL(self, "txt_dose")
        self.txt_dose.SetValue("%.2f" % (voi.get_dose()))

        self.check_target = XRCCTRL(self, "check_target")
        self.check_target.SetValue(voi.is_target())
        self.check_target.Bind(wx.EVT_CHECKBOX, self.on_check_target_changed)

        self.check_oar = XRCCTRL(self, "check_oar")
        self.check_oar.SetValue(voi.is_oar())
        self.check_oar.Bind(wx.EVT_CHECKBOX, self.on_check_oar_changed)

        self.txt_max_dose_fraction = XRCCTRL(self, "txt_max_dose_fraction")
        self.txt_max_dose_fraction.SetValue("%.2f" % (voi.get_max_dose_fraction()))

        self.txt_max_dose_fraction.Enable(False)
        self.txt_dose.Enable(False)
        if voi.is_target():
            self.check_oar.Enable(False)
            self.txt_dose.Enable(True)

        if voi.is_oar():
            self.txt_max_dose_fraction.Enable(True)
            self.check_target.Enable(False)

        self.txt_hu_value = XRCCTRL(self, "txt_hu_value")
        self.txt_hu_offset = XRCCTRL(self, "txt_hu_offset")
        if not voi.get_hu_value() is None:
            self.txt_hu_value.SetValue("%d" % voi.get_hu_value())
        if not voi.get_hu_offset() is None:
            self.txt_hu_offset.SetValue("%d" % voi.get_hu_offset())

        self.drop_projectile = XRCCTRL(self, "drop_projectile")
        self.drop_projectile.Append("H")
        self.drop_projectile.Append("C")

        self.txt_dose_percent = XRCCTRL(self, "txt_dose_percent")
        wx.EVT_BUTTON(self, XRCID('btn_set_dosepercent'), self.set_dose_percent)
        wx.EVT_CHOICE(self, XRCID('drop_projectile'), self.on_projectile_changed)

    def on_projectile_changed(self, evt):
        projectile = self.drop_projectile.GetStringSelection()
        dose_percent = self.voi.get_dose_percent(projectile)
        if dose_percent is None:
            self.txt_dose_percent.SetValue("")
        else:
            self.txt_dose_percent.SetValue("%d" % dose_percent)

    def set_dose_percent(self, evt):
        if not self.drop_projectile.GetStringSelection() == "":
            self.voi.set_dose_percent(self.drop_projectile.GetStringSelection(), self.txt_dose_percent.GetValue())

    def on_check_target_changed(self, evt):
        if evt.Checked():
            self.check_oar.Enable(False)
            self.txt_dose.Enable(True)
        else:
            self.check_oar.Enable(True)
            self.txt_dose.Enable(False)

    def on_check_oar_changed(self, evt):
        if evt.Checked():
            self.txt_max_dose_fraction.Enable(True)
            self.check_target.Enable(False)
        else:
            self.check_target.Enable(True)
            self.txt_max_dose_fraction.Enable(False)

    def save_and_close(self, evt):
        voi = self.voi
        voi.set_dose(self.txt_dose.GetValue())
        if voi.is_target() is not self.check_target.IsChecked():
            voi.toogle_target()
        if voi.is_oar() is not self.check_oar.IsChecked():
            voi.toogle_oar()
        voi.set_max_dose_fraction(self.txt_max_dose_fraction.GetValue())
        voi.set_hu_offset(self.txt_hu_offset.GetValue())
        voi.set_hu_value(self.txt_hu_value.GetValue())

        self.Close()

    def close(self, evt):
        self.Close()
