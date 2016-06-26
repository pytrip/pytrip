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


class PlanDialog(wx.Dialog):
    def __init__(self):
        pre = wx.PreDialog()
        self.PostCreate(pre)
        pub.subscribe(self.patient_data_updated, "patient.loaded")
        pub.sendMessage("patient.request", {})

    def Init(self, plan):
        self.plan = plan

        self.btn_ok = XRCCTRL(self, 'btn_ok')
        wx.EVT_BUTTON(self, XRCID('btn_ok'), self.save_and_close)

        self.btn_cancel = XRCCTRL(self, 'btn_close')
        wx.EVT_BUTTON(self, XRCID('btn_close'), self.close)

        self.init_general()
        self.init_trip_panel()
        self.init_opt_panel()
        self.init_calculation_panel()
        self.init_files_panel()
        self.init_advanved_dose()

    def patient_data_updated(self, msg):
        self.data = msg.data

    def init_files_panel(self):
        self.txt_ddd = XRCCTRL(self, "txt_ddd")
        self.txt_ddd.SetValue(self.plan.get_ddd_folder())

        self.txt_spc = XRCCTRL(self, "txt_spc")
        self.txt_spc.SetValue(self.plan.get_spc_folder())

        self.txt_sis = XRCCTRL(self, "txt_sis")
        self.txt_sis.SetValue(self.plan.get_sis_file())

        wx.EVT_BUTTON(self, XRCID("btn_ddd"), self.on_btn_ddd_clicked)
        wx.EVT_BUTTON(self, XRCID("btn_spc"), self.on_btn_spc_clicked)
        wx.EVT_BUTTON(self, XRCID("btn_sis"), self.on_btn_sis_clicked)

    def init_general(self):
        self.drop_res_tissue_type = XRCCTRL(self, "drop_res_tissue_type")
        rbe_list = self.data.get_rbe()
        for rbe in rbe_list.get_rbe_list():
            self.drop_res_tissue_type.Append(rbe.get_name())
        self.select_drop_by_value(self.drop_res_tissue_type, self.plan.get_res_tissue_type())
        self.drop_target_tissue_type = XRCCTRL(self, "drop_target_tissue_type")
        for rbe in rbe_list.get_rbe_list():
            self.drop_target_tissue_type.Append(rbe.get_name())
        self.select_drop_by_value(self.drop_target_tissue_type, self.plan.get_target_tissue_type())

    def select_drop_by_value(self, drop, value):
        for i, item in enumerate(drop.GetItems()):
            if item == value:
                drop.SetSelection(i)

    def init_calculation_panel(self):
        self.check_phys_dose = XRCCTRL(self, "check_phys_dose")
        self.check_phys_dose.SetValue(self.plan.get_out_phys_dose())

        self.check_bio_dose = XRCCTRL(self, "check_bio_dose")
        self.check_bio_dose.SetValue(self.plan.get_out_bio_dose())

        self.check_dose_mean_let = XRCCTRL(self, "check_mean_let")
        self.check_dose_mean_let.SetValue(self.plan.get_out_dose_mean_let())

        self.check_field = XRCCTRL(self, "check_field")
        self.check_field.SetValue(self.plan.get_out_field())

    def init_opt_panel(self):
        self.txt_iterations = XRCCTRL(self, "txt_iterations")
        self.txt_iterations.SetValue("%d" % self.plan.get_iterations())

        self.txt_eps = XRCCTRL(self, "txt_eps")
        self.txt_eps.SetValue("%f" % self.plan.get_eps())

        self.txt_geps = XRCCTRL(self, "txt_geps")
        self.txt_geps.SetValue("%f" % self.plan.get_geps())

        self.drop_opt_method = XRCCTRL(self, "drop_opt_method")
        self.select_drop_by_value(self.drop_opt_method, self.plan.get_opt_method())

        self.drop_opt_principle = XRCCTRL(self, "drop_opt_principle")
        self.select_drop_by_value(self.drop_opt_principle, self.plan.get_opt_princip())

        self.drop_dose_alg = XRCCTRL(self, "drop_dose_alg")
        self.select_drop_by_value(self.drop_dose_alg, self.plan.get_dose_algorithm())

        self.drop_bio_alg = XRCCTRL(self, "drop_bio_alg")
        self.select_drop_by_value(self.drop_bio_alg, self.plan.get_dose_algorithm())

        self.drop_opt_alg = XRCCTRL(self, "drop_opt_alg")
        self.select_drop_by_value(self.drop_opt_alg, self.plan.get_opt_algorithm())

    def init_trip_panel(self):
        self.drop_location = XRCCTRL(self, "drop_location")
        if self.plan.is_remote():
            self.drop_location.SetSelection(1)

        self.txt_working_dir = XRCCTRL(self, "txt_working_dir")
        self.txt_working_dir.SetValue(self.plan.get_working_dir())

        wx.EVT_BUTTON(self, XRCID('btn_working_dir'), self.on_browse_working_dir)

        self.txt_username = XRCCTRL(self, "txt_username")
        self.txt_username.SetValue(self.plan.get_username())

        self.txt_password = XRCCTRL(self, "txt_password")
        self.txt_password.SetValue(self.plan.get_password())

        self.txt_server = XRCCTRL(self, "txt_server")
        self.txt_server.SetValue(self.plan.get_server())

    def init_advanved_dose(self):
        self.drop_projectile = XRCCTRL(self, "drop_projectile")
        self.drop_projectile.Append("H")
        self.drop_projectile.Append("C")

        self.txt_dose_percent = XRCCTRL(self, "txt_dose_percent")
        wx.EVT_BUTTON(self, XRCID('btn_set_dosepercent'), self.set_dose_percent)
        wx.EVT_CHOICE(self, XRCID('drop_projectile'), self.on_projectile_changed)

    def on_projectile_changed(self, evt):
        projectile = self.drop_projectile.GetStringSelection()
        dose_percent = self.plan.get_dose_percent(projectile)
        if dose_percent is None:
            self.txt_dose_percent.SetValue("")
        else:
            self.txt_dose_percent.SetValue("%d" % dose_percent)

    def set_dose_percent(self, evt):
        if not self.drop_projectile.GetStringSelection() == "":
            self.plan.set_dose_percent(self.drop_projectile.GetStringSelection(), self.txt_dose_percent.GetValue())

    def on_browse_working_dir(self, evt):
        dlg = wx.DirDialog(
            self,
            defaultPath=self.txt_working_dir.GetValue(),
            message="Choose the folder pytripgui should use as working directory")
        if dlg.ShowModal() == wx.ID_OK:
            self.txt_working_dir.SetValue(dlg.GetPath())

    def on_btn_ddd_clicked(self, evt):
        dlg = wx.DirDialog(
            self,
            defaultPath=self.txt_ddd.GetValue(),
            message="Choose folder where ddd files are located")
        if dlg.ShowModal() == wx.ID_OK:
            self.txt_ddd.SetValue(dlg.GetPath())

    def on_btn_sis_clicked(self, evt):
        dlg = wx.FileDialog(
            self,
            defaultFile=self.txt_sis.GetValue(),
            message="Choose sis file")
        if dlg.ShowModal() == wx.ID_OK:
            self.txt_sis.SetValue(dlg.GetPath())

    def on_btn_spc_clicked(self, evt):
        dlg = wx.DirDialog(
            self,
            defaultPath=self.txt_spc.GetValue(),
            message="Choose folder where spc files are located")
        if dlg.ShowModal() == wx.ID_OK:
            self.txt_spc.SetValue(dlg.GetPath())

    def save_and_close(self, evt):
        self.plan.set_res_tissue_type(self.drop_res_tissue_type.GetStringSelection())
        self.plan.set_target_tissue_type(self.drop_target_tissue_type.GetStringSelection())
        if self.drop_location.GetSelection() is 0:
            self.plan.set_remote_state(False)
        else:
            self.plan.set_remote_state(True)
        self.plan.set_working_dir(self.txt_working_dir.GetValue())
        self.plan.set_server(self.txt_server.GetValue())
        self.plan.set_username(self.txt_username.GetValue())
        self.plan.set_password(self.txt_password.GetValue())

        self.plan.set_iterations(self.txt_iterations.GetValue())
        self.plan.set_eps(self.txt_eps.GetValue())
        self.plan.set_geps(self.txt_geps.GetValue())
        self.plan.set_opt_method(self.drop_opt_method.GetStringSelection())
        self.plan.set_opt_princip(self.drop_opt_principle.GetStringSelection())
        self.plan.set_dose_algorithm(self.drop_dose_alg.GetStringSelection())
        self.plan.set_bio_algorithm(self.drop_bio_alg.GetStringSelection())
        self.plan.set_opt_algorithm(self.drop_opt_alg.GetStringSelection())

        self.plan.set_out_phys_dose(self.check_phys_dose.GetValue())
        self.plan.set_out_bio_dose(self.check_bio_dose.GetValue())
        self.plan.set_out_dose_mean_let(self.check_dose_mean_let.GetValue())
        self.plan.set_out_field(self.check_field.GetValue())

        self.plan.set_ddd_folder(self.txt_ddd.GetValue())
        self.plan.set_spc_folder(self.txt_ddd.GetValue())
        self.plan.set_sis_file(self.txt_sis.GetValue())

        self.Close()

    def close(self, evt):
        self.Close()
