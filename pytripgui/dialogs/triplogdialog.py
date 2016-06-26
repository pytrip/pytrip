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


class TripLogDialog(wx.Dialog):
    def __init__(self):
        pre = wx.PreDialog()
        self.PostCreate(pre)

    def Init(self, tripexecuter):
        self.tripexecuter = tripexecuter
        self.tripexecuter.add_log_listener(self)
        self.txt_log = XRCCTRL(self, "txt_log")

        wx.EVT_BUTTON(self, XRCID("btn_ok"), self.close)
        self.btn_ok = XRCCTRL(self, "btn_ok")
        self.btn_ok.Enable(False)
        self.check_close = XRCCTRL(self, "check_close")

    def close(self, evt):
        self.Close()

    def finish(self):
        self.btn_ok.Enable(True)
        if self.check_close.IsChecked():
            self.Close()

    def write(self, txt):
        self.txt_log.AppendText("%s\n" % txt)
