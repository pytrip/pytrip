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
import wx, wx.lib.dialogs

from pytrip.error import *
import sys, traceback
from wx.xrc import *
from leftmenu import *
from settings import *
from plugin import *
from panels.plotpanel import *
from panels.dvh import *
from tripexecparser import *
import threading
import os
import util
from data import *

if getattr(sys, 'frozen', False):
    from wx.lib.pubsub import pub
    from wx.lib.pubsub import setuparg1
else:
    try:
        from wx.lib.pubsub import Publisher as pub
    except:
        from wx.lib.pubsub import setuparg1
        from wx.lib.pubsub import pub

import gc


class FileDropTarget(wx.FileDropTarget):
    def __init__(self, obj):
        wx.FileDropTarget.__init__(self)
        self.obj = obj

    def OnDropFiles(self, x, y, filenames):
        if os.path.splitext(filenames[0])[1] == ".pyt":
            self.obj.load_pyt(filenames[0])


class MainFrame(wx.Frame):
    def __init__(self):
        pre = wx.PreFrame()
        # the Create step is done by XRC.
        self.PostCreate(pre)

    def Init(self, res):
        self.leftmenu_panel = XRCCTRL(self, "leftmenu_panel")
        self.main_notebook = XRCCTRL(self, "main_notebook")
        self.statusbar = XRCCTRL(self, "statusbar")
        wx.EVT_NOTEBOOK_PAGE_CHANGED(self, XRCID("main_notebook"), self.main_notebook_active_page_changed)

        sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.leftmenu = LeftMenuTree(self.leftmenu_panel, -1, size=(200, -1),
                                     style=wx.ALL | wx.EXPAND | wx.TR_DEFAULT_STYLE | wx.TR_EDIT_LABELS)
        sizer.Add(self.leftmenu, 1, wx.EXPAND, 0)
        self.leftmenu_panel.SetSizer(sizer)
        self.data = PytripData()
        self.res = res
        self.bind_menu()
        self.bind_toolbar()
        self.load_settings()

        pub.subscribe(self.on_patient_load, "patient.load")
        pub.subscribe(self.on_import_path_change, "general.import")
        pub.subscribe(self.statusbar_updated, "statusbar.update")
        pub.sendMessage("settings.value.request", "general.import.dicom_path")
        pub.sendMessage("settings.value.request", "general.import.voxelplan_path")
        pub.sendMessage("settings.value.request", "general.import.tripexec_path")

        pub.subscribe(self.load_dialog, "gui")

        self.res = XmlResource(util.get_resource_path('panels.xrc'))

    def statusbar_updated(self, msg):
        self.statusbar.SetStatusText(msg.data["text"], msg.data["number"])

    def main_notebook_active_page_changed(self, evt):
        self.bind_default_toolbar()
        self.statusbar.SetStatusText("", 0)
        self.statusbar.SetStatusText("", 1)
        self.statusbar.SetStatusText("", 2)

    def load_dialog(self, msg):
        dialogs = {"field": "FieldDialog", "tripplan": "PlanDialog", "tripvoi": "TripVoiDialog",
                   "dose": "DoseDialog", "triplog": "TripLogDialog", "wait": "ProgressDialog",
                   "tripexport": "TripExportDialog", "tripcubeexport": "TripExportCubeDialog"}
        panels = {"dvh": DVHPanel, "lvh": LVHPanel}
        if msg.topic[2] == "open":
            if msg.topic[1] in dialogs.keys():
                pytripDialog = self.res.LoadDialog(self, dialogs[msg.topic[1]])
                pytripDialog.Init(msg.data)
                self.Enable(False)
                pytripDialog.ShowModal()
                self.Enable(True)
                self.dialog = pytripDialog
            elif msg.topic[1] in panels.keys():
                panel = panels[msg.topic[1]](self.main_notebook)
                panel.Init(msg.data)
                self.main_notebook.AddPage(panel, panel.get_title(), select=True)
        elif msg.topic[2] == "close":
            self.dialog.Close()

    def ini_plugins(self):
        self.plugins = PluginManager()
        self.plugins.load_modules()

        self.menuitem_import = XRCCTRL(self, "top_menu")
        import_plugins = self.plugins.get_plugins_by_type("import")
        for plugin in import_plugins:
            id = wx.NewId()
            self.import_menu.Append(id, plugin.pluginProperties()["menuname"])
            plug = plugin.plugin(self)
            wx.EVT_MENU(self, id, plug.pluginMenu)

    def on_patient_load(self, msg):
        self.clean_up()

        self.data = msg.data
        self.load_notetab_panels()
        pub.sendMessage("patient.loaded.ini", self.data)
        self.top_menu.Enable(XRCID("menuitem_view_dvh"), True)
        self.top_menu.Enable(XRCID("menuitem_view_lvh"), True)

        self.bind_default_toolbar()

    def load_notetab_panels(self):
        self.main_notebook.DeleteAllPages()
        plot = PlotPanel(self.main_notebook)
        plot.Init()
        self.main_notebook.AddPage(plot, "2d Plot")

    def on_import_path_change(self, msg):
        data = msg.data
        if data is None:
            data = ""
        setattr(self, msg.topic[2], data)

    def load_settings(self):
        self.settings_manager = SettingsManager()
        self.settings_manager.load_settings()

    def bind_menu(self):
        self.top_menu = self.res.LoadMenuBar("top_menu")
        self.SetMenuBar(self.top_menu)
        self.import_menu = self.top_menu.FindItemById(XRCID("submenu_import")).GetSubMenu()
        self.export_menu = self.top_menu.FindItemById(XRCID("submenu_export"))

        # ~ self.export_menu.Enable()
        wx.EVT_MENU(self, XRCID("menuitem_openpatient"), self.open_patient_load_dialog)
        wx.EVT_MENU(self, XRCID("menuitem_voxelplan"), self.voxelplan_load_dialog)
        wx.EVT_MENU(self, XRCID("menuitem_importexec"), self.tripexec_load_dialog)
        wx.EVT_MENU(self, XRCID("menuitem_createcube"), self.createcube_load_dialog)
        wx.EVT_MENU(self, XRCID("menuitem_createstructure"), self.createstructure_load_dialog)
        wx.EVT_MENU(self, XRCID("menuitem_importexec"), self.tripexec_load_dialog)
        wx.EVT_MENU(self, XRCID("menuitem_quit"), self.close)
        wx.EVT_MENU(self, XRCID("menuitem_save"), self.save)
        wx.EVT_MENU(self, XRCID("menuitem_saveas"), self.saveas)
        wx.EVT_MENU(self, XRCID("menuitem_load"), self.load)
        wx.EVT_MENU(self, XRCID("menuitem_license"), self.view_licence)
        wx.EVT_MENU(self, XRCID("menuitem_about"), self.view_about)

        wx.EVT_MENU(self, XRCID("menuitem_view_dvh"), self.view_dvh)
        wx.EVT_MENU(self, XRCID("menuitem_view_lvh"), self.view_lvh)
        self.top_menu.Enable(XRCID("menuitem_view_dvh"), False)
        self.top_menu.Enable(XRCID("menuitem_view_lvh"), False)

    def view_dvh(self, evt):
        pub.sendMessage("gui.dvh.open", None)

    def view_lvh(self, evt):
        pub.sendMessage("gui.lvh.open", None)

    def view_about(self, evt):
        info = wx.AboutDialogInfo()
        with open(os.path.join(util.get_main_dir(), "LICENSE"), "rU") as fp:
            licence = fp.read()
        description = "Bla bla description"
        info.SetName('PyTRiP')
        info.SetVersion('0.1')
        info.SetDescription(description)
        info.SetCopyright('(C) 2012 - 2013 Aarhus Particle Therapy Group')
        info.SetWebSite('https://svn.nfit.au.dk/trac/pytrip')
        info.SetLicence(licence)
        info.AddDeveloper('Jakob Toftegaard')
        info.AddDeveloper('Niels Bassler')
        wx.AboutBox(info)

    def view_licence(self, evt):
        with open(os.path.join(util.get_main_dir(), "LICENSE"), "rU") as fp:
            msg = fp.read()
        dlg = wx.lib.dialogs.ScrolledMessageDialog(self, msg,
                                                   "PyTRiP License")
        dlg.ShowModal()
        dlg.Destroy()

    def load(self, evt):
        dlg = wx.FileDialog(
            self,
            wildcard="PyTRiP project files (*.pyt)|*.pyt",
            message="Choose PyTRiP project file")
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.load_pyt(path)
            self.savepath = path

    def load_pyt(self, path):
        self.data = PytripData()
        self.data.load(path)

    def save(self, evt):
        if not hasattr(self, "savepath"):
            self.saveas(evt)
        else:
            self.data.save(self.savepath)

    def saveas(self, evt):
        dlg = wx.FileDialog(
            self,
            wildcard="PyTRiP project files (*.pyt)|*.pyt",
            message="Save Project",
            style=wx.FD_SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.savepath = os.path.splitext(path)[0] + ".pyt"
            self.data.save(self.savepath)

    def bind_toolbar(self):
        self.toolbar = self.CreateToolBar()
        self.bind_default_toolbar()

    def bind_default_toolbar(self):
        self.toolbar.ClearTools()
        id = wx.NewId()
        self.toolbar.AddLabelTool(id, '', wx.Bitmap(get_resource_path('open.png')))
        wx.EVT_MENU(self, id, self.open_patient_load_dialog)
        page = self.main_notebook.GetCurrentPage()
        if page is not None:
            if not hasattr(page, "is_closeable") or page.is_closeable:
                id = wx.NewId()
                self.toolbar.AddLabelTool(id, '', wx.Bitmap(get_resource_path('delete.png')))
                wx.EVT_MENU(self, id, self.close_panel)
            if hasattr(page, "get_figure"):
                id = wx.NewId()
                self.toolbar.AddLabelTool(id, '', wx.Bitmap(get_resource_path('picture.png')))
                wx.EVT_MENU(self, id, self.save_as_picture)
        self.toolbar.AddSeparator()
        if hasattr(page, "set_toolbar"):
            page.set_toolbar(self.toolbar)

    def close_panel(self, evt):
        self.main_notebook.DeletePage(self.main_notebook.GetSelection())
        self.bind_default_toolbar()

    def save_as_picture(self, evt):
        page = self.main_notebook.GetCurrentPage()
        dlg = wx.FileDialog(
            self,
            message="Save Picture",
            style=wx.FD_SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            ext = os.path.splitext(path)[1]
            if len(ext) is 1:
                path = os.path.splitext(path)[0] + ".png"
            else:
                if not ext.lower() in [".eps", ".jpg", ".jpeg", ".png", ".gif"]:
                    return
            page.get_figure().savefig(path, bbox_inches=0, facecolor=page.get_figure().get_facecolor(), edgecolor=None)

    def createcube_load_dialog(self, evt):
        dia = self.res.LoadDialog(self, "CreateCubeDialog")
        dia.Init(self)
        self.Enable(False)
        dia.ShowModal()
        self.Enable(True)

    def createstructure_load_dialog(self, evt):
        dia = self.res.LoadDialog(self, "CreateStructureDialog")
        dia.Init(self)
        self.Enable(False)
        dia.ShowModal()
        self.Enable(True)

    def tripexec_load_dialog(self, evt):
        dlg = wx.FileDialog(
            self,
            defaultFile=self.tripexec_path,
            wildcard="TRiP Exec File (*.exec)|*.exec|",
            message="Choose TRiP Exec File")
        if dlg.ShowModal() == wx.ID_OK:
            data = PytripData()
            path = dlg.GetPath()
            pub.sendMessage("settings.value.updated", {"general.import.tripexec_path": path})
            data.load_trip_exec(path)

    def voxelplan_load_dialog(self, evt):
        dlg = wx.FileDialog(
            self,
            defaultFile=self.voxelplan_path,
            wildcard="Voxelplan headerfile (*.hed)|*.hed|",
            message="Choose headerfile")
        if dlg.ShowModal() == wx.ID_OK:
            data_obj = PytripData()
            path = dlg.GetPath()
            pub.sendMessage("settings.value.updated", {"general.import.voxelplan_path": path})
            data_obj.load_from_voxelplan(path)

    def open_patient_load_dialog(self, evt):
        dlg = wx.DirDialog(
            self,
            defaultPath=self.dicom_path,
            message="Choose the folder where the dicom files are stored")
        if dlg.ShowModal() == wx.ID_OK:
            data_obj = PytripData()
            path = dlg.GetPath()
            pub.sendMessage("settings.value.updated", {"general.import.dicom_path": path})
            data_obj.load_from_dicom(path)

    def clean_up(self):
        gc.collect()

    def close(self, evt):
        self.Close()


class pytripgui(wx.App):
    def OnInit(self):
        wx.InitAllImageHandlers()
        wx.GetApp().SetAppName("pytrip")
        # Load the XRC file for our gui resources
        self.res = XmlResource(util.get_resource_path('main.xrc'))
        pytripFrame = self.res.LoadFrame(None, 'MainFrame')
        font = wx.SystemSettings_GetFont(wx.SYS_DEFAULT_GUI_FONT)
        pytripFrame.SetFont(font)
        pytripFrame.Init(self.res)
        dt1 = FileDropTarget(pytripFrame)
        pytripFrame.SetDropTarget(dt1)
        self.SetTopWindow(pytripFrame)
        pytripFrame.Centre()
        pytripFrame.Show()
        return 1


def handleInputException(exc_type, exc_value, exc_traceback):
    if exc_type is InputError:
        dlg = wx.MessageDialog(None, str(exc_value), 'Input Error', wx.OK | wx.ICON_ERROR)
    else:
        err_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        dlg = wx.MessageDialog(None, err_msg, 'Termination dialog', wx.OK | wx.ICON_ERROR)
    dlg.ShowModal()
    dlg.Destroy()


sys.excepthook = handleInputException


def start():
    app = pytripgui(0)
    app.MainLoop()


if __name__ == '__main__':
    start()
