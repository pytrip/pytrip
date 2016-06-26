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
import pdb

if getattr(sys, 'frozen', False):
    from wx.lib.pubsub import pub
    from wx.lib.pubsub import setuparg1
else:
    try:
        from wx.lib.pubsub import Publisher as pub
    except:
        from wx.lib.pubsub import setuparg1
        from wx.lib.pubsub import pub

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
import matplotlib

matplotlib.interactive(True)
import matplotlib.pyplot as plt
from pytrip.guiutil import PlotUtil
from pytripgui.util import *
import numpy as np
import threading
import time


def cmap_discretize(cmap, N):
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


class PlotPanel(wx.Panel):
    zoom_levels = [100.0, 110.0, 125.0, 150.0, 200.0, 250.0, 300.0, 400.0]
    dose_contour_levels = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 98 - 0, 100.0, 102.0]

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.is_closeable = False
        self.parent = parent
        self.active_plan = None
        self.plot_mouse_action = None

        pub.subscribe(self.on_patient_loaded, "patient.loaded.ini")
        pub.subscribe(self.set_active_image, "plot.image.active_id")
        pub.subscribe(self.voi_changed, "voi.selection_changed")
        pub.subscribe(self.plan_changed, "plan.active.changed")
        pub.subscribe(self.plan_field_changed, "plan.field.selection_changed")
        pub.subscribe(self.plan_dose_changed, "plan.dose.active_changed")
        pub.subscribe(self.plan_dose_removed, "plan.dose.removed")
        pub.subscribe(self.plan_let_added, "plan.let.added")
        pub.subscribe(self.plan_let_removed, "plan.let.removed")
        pub.subscribe(self.target_dose_changed, "plan.dose.target_dose_changed")
        self.plotmode = "Transversal"

    def __del__(self):
        pub.unsubscribe(self.on_patient_loaded, "patient.loaded.ini")
        pub.unsubscribe(self.set_active_image, "plot.image.active_id")
        pub.unsubscribe(self.voi_changed, "voi.selection_changed")
        pub.unsubscribe(self.plan_changed, "plan.active.changed")
        pub.unsubscribe(self.plan_field_changed, "plan.field.selection_changed")
        pub.unsubscribe(self.plan_dose_changed, "plan.dose.active_changed")
        pub.unsubscribe(self.plan_dose_removed, "plan.dose.removed")
        pub.unsubscribe(self.plan_let_added, "plan.let.added")
        pub.unsubscribe(self.plan_let_removed, "plan.let.removed")
        pub.unsubscribe(self.target_dose_changed, "plan.dose.target_dose_changed")

    def target_dose_changed(self, msg):
        self.Draw()

    def Init(self):
        self.plotutil = PlotUtil()
        self.figure = Figure(None, 100)
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        # ~ self.canvas.SetDoubleBuffered(True)
        self.clear()
        self.plotutil.set_draw_in_gui(True)
        self.figure.set_frameon(True)
        rect = self.figure.patch
        rect.set_facecolor('black')

    def plan_dose_changed(self, msg):
        if msg.data["plan"] is self.active_plan:
            self.plotutil.set_dose(msg.data["dose"].get_dosecube())
            self.Draw()

    def plan_field_changed(self, msg):
        self.Draw()

    def plan_dose_removed(self, msg):
        if msg.data["plan"] is self.active_plan:
            self.plotutil.set_dose(None)
            self.Draw()

    def plan_let_added(self, msg):
        if msg.data["plan"] is self.active_plan:
            self.plotutil.set_let(msg.data["let"])
            self.Draw()

    def plan_let_removed(self, msg):
        if msg.data["plan"] is self.active_plan:
            self.plotutil.set_let(None)
            self.Draw()

    def plan_changed(self, msg):
        self.active_plan = msg.data
        if self.active_plan is None:
            self.plotutil.set_plan(None)
            self.plotutil.set_dose(None)
            self.plotutil.set_let(None)
        else:
            self.plotutil.set_plan(self.active_plan)
            doseobj = self.active_plan.get_dose()

            if doseobj is not None:
                self.plotutil.set_dose(doseobj.get_dosecube())
            else:
                self.plotutil.set_dose(None)
            self.plotutil.set_let(self.active_plan.get_let())
        self.Draw()

    def set_toolbar(self, toolbar):
        id = wx.NewId()
        selector = wx.Choice(toolbar, id)
        selector.Append("Transversal")
        selector.Append("Sagital")
        selector.Append("Coronal")
        idx = selector.FindString(self.plotmode)
        selector.Select(idx)

        toolbar.AddControl(selector)
        wx.EVT_CHOICE(selector, id, self.plot_mode_changed)

        id = wx.NewId()
        self.zoom_in_btn = toolbar.AddLabelTool(id, '', wx.Bitmap(get_resource_path('zoom_in.png')))
        wx.EVT_MENU(toolbar, id, self.zoom_in)

        id = wx.NewId()
        self.zoom_out_btn = toolbar.AddLabelTool(id, '', wx.Bitmap(get_resource_path('zoom_out.png')))
        wx.EVT_MENU(toolbar, id, self.zoom_out)

    def zoom_buttons_visible(self):
        zoom_idx = self.zoom_levels.index(self.plotutil.get_zoom())
        self.zoom_in_btn.Enable(True)
        self.zoom_out_btn.Enable(True)

        if len(self.zoom_levels) == zoom_idx + 1:
            self.zoom_in_btn.Enable(False)
        if zoom_idx == 0:
            self.zoom_out_btn.Enable(False)

    def zoom_in(self, evt):
        zoom_idx = self.zoom_levels.index(self.plotutil.get_zoom())
        zoom_idx += 1
        if len(self.zoom_levels) > zoom_idx:
            zoom = self.zoom_levels[zoom_idx]
            self.plotutil.set_zoom(zoom)
            self.zoom_buttons_visible()
            self.Draw()

    def zoom_out(self, evt):
        zoom_idx = self.zoom_levels.index(self.plotutil.get_zoom())
        zoom_idx -= 1
        if zoom_idx >= 0:
            zoom = self.zoom_levels[zoom_idx]
            self.plotutil.set_zoom(zoom)
            self.zoom_buttons_visible()
            self.Draw()

    def plot_mode_changed(self, evt):
        self.plotmode = evt.GetString()
        self.plotutil.set_plot_plan(self.plotmode)
        self.image_idx = int(self.plotutil.get_images_count() / 2)
        self.clear()
        self.Draw()

    def clear(self):
        self.figure.clear()
        self.subplot = self.figure.add_subplot(111)
        self.plotutil.set_figure(self.subplot)

    def voi_changed(self, msg):
        voi = msg.data
        if voi.is_selected():
            self.plotutil.add_voi(voi.voxelplan_voi)
        else:
            self.plotutil.remove_voi(voi.voxelplan_voi)
        self.Draw()

    def set_active_image(self, msg):
        if self.plotmode == "Transversal":
            self.image_idx = msg.data
            self.Draw()

    def on_patient_loaded(self, msg):
        self.data = msg.data
        ctx = self.data.get_images().get_voxelplan()

        self.plotutil.set_ct(ctx)

        self.image_idx = int(ctx.dimz / 2)
        self.setSize()
        self.bind_keys()

    def get_figure(self):
        return self.figure

    def bind_keys(self):
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)
        self.Bind(wx.EVT_ENTER_WINDOW, self.on_mouse_enter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_mouse_leave)

        self.Bind(wx.EVT_SIZE, self.on_size)
        self.canvas.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move_plot)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press_plot)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_action_ended)
        self.canvas.mpl_connect('figure_leave_event', self.on_mouse_action_ended)

    def on_mouse_press_plot(self, evt):
        if evt.button is 3:
            pos = evt.guiEvent.GetPosition()
            standard = True
            if hasattr(self.plotutil, "contrast_bar"):
                bar = self.plotutil.contrast_bar
                if evt.inaxes is bar.ax:
                    menu = self.right_click_contrast()
                    standard = False
            if hasattr(self.plotutil, "dose_bar"):
                bar = self.plotutil.dose_bar
                if evt.inaxes is bar.ax:
                    menu = self.right_click_dose()
                    standard = False

            if standard:
                menu = self.normal_right_click_menu()

            wx.CallAfter(self.show_menu, menu);
            if self.canvas.HasCapture():
                self.canvas.ReleaseMouse()

        elif evt.button is 1:
            if hasattr(self.plotutil, "contrast_bar"):
                bar = self.plotutil.contrast_bar
                if evt.inaxes is bar.ax:
                    if evt.ydata >= 0.50:
                        self.plot_mouse_action = "contrast_top"
                    else:
                        self.plot_mouse_action = "contrast_bottom"
            if hasattr(self.plotutil, "dose_bar"):
                bar = self.plotutil.dose_bar
                if evt.inaxes is bar.ax:
                    if evt.ydata >= 0.50:
                        self.plot_mouse_action = "dose_top"
                    else:
                        self.plot_mouse_action = "dose_bottom"
            if hasattr(self.plotutil, "let_bar"):
                bar = self.plotutil.let_bar
                if evt.inaxes is bar.ax:
                    if evt.ydata >= 0.50:
                        self.plot_mouse_action = "let_top"
                    else:
                        self.plot_mouse_action = "let_bottom"

        self.mouse_pos_ini = [evt.x, evt.y]
        evt.guiEvent.Skip()

    def show_menu(self, menu):
        self.PopupMenu(menu)

    def on_mouse_action_ended(self, evt):
        self.plot_mouse_action = None

    def on_mouse_move_plot(self, evt):
        pos = [evt.x, evt.y]
        if self.plot_mouse_action is not None:
            step = [pos[0] - self.mouse_pos_ini[0], pos[1] - self.mouse_pos_ini[1]]
            if self.plot_mouse_action == "contrast_top":
                contrast = self.plotutil.get_contrast()
                stepsize = np.log(contrast[1] - contrast[0])
                contrast[1] -= stepsize * step[1]
                self.plotutil.set_contrast(contrast)
            elif self.plot_mouse_action == "contrast_bottom":
                contrast = self.plotutil.get_contrast()
                stepsize = np.log(contrast[1] - contrast[0])
                contrast[0] -= stepsize * step[1]
                self.plotutil.set_contrast(contrast)
            elif self.plot_mouse_action == "dose_top":
                dose = self.plotutil.get_min_max_dose()
                dose[1] -= 0.30 * step[1]
                self.plotutil.set_dose_min_max(dose)
            elif self.plot_mouse_action == "dose_bottom":
                dose = self.plotutil.get_min_max_dose()
                dose[0] -= 0.30 * step[1]
                self.plotutil.set_dose_min_max(dose)
            elif self.plot_mouse_action == "let_top":
                let = self.plotutil.get_min_max_let()
                let[1] -= 0.30 * step[1]
                self.plotutil.set_let_min_max(let)
            elif self.plot_mouse_action == "let_bottom":
                let = self.plotutil.get_min_max_let()
                let[0] -= 0.30 * step[1]
                self.plotutil.set_let_min_max(let)
            self.Draw()
        elif evt.button == 1 and evt.inaxes is self.plotutil.fig_ct.axes:
            if not None in self.mouse_pos_ini:
                step = [pos[0] - self.mouse_pos_ini[0], pos[1] - self.mouse_pos_ini[1]]
                if self.plotutil.move_center(step):
                    self.Draw()

        self.mouse_pos_ini = [evt.x, evt.y]
        if hasattr(self.plotutil, "fig_ct") and evt.inaxes is self.plotutil.fig_ct.axes:
            point = self.plotutil.pixel_to_pos([round(evt.xdata), round(evt.ydata)])

            text = "X: %.2f mm Y: %.2f mm / X: %d px Y: %d px" % (point[1][0], point[1][1], point[0][0], point[0][1])
            pub.sendMessage("statusbar.update", {"number": 1, "text": text})
            dim = self.data.get_image_dimensions()
            if self.plotmode == "Transversal":
                pos = [round(evt.xdata), round(evt.ydata), self.image_idx]
            elif self.plotmode == "Sagital":
                pos = [dim[0] - round(evt.xdata), self.image_idx, dim[2] - round(evt.ydata)]
            elif self.plotmode == "Coronal":
                pos = [self.image_idx, dim[1] - round(evt.xdata), dim[2] - round(evt.ydata)]
            try:
                ct_value = self.data.get_image_cube()[pos[2], pos[1], pos[0]]
                text = "Value: %.1f" % (ct_value)
                plan = self.active_plan
                if plan is not None:
                    dose = plan.get_dose_cube()
                    if dose is not None:
                        dose_value = dose[pos[2], pos[1], pos[0]]
                        target_dose = plan.get_dose().get_dose()
                        if not target_dose == 0.0:
                            dose_value *= target_dose / 1000
                            text += " / Dose: %.1f Gy" % (float(dose_value))
                        else:
                            dose_value /= 10
                            text += " / Dose: %.1f %%" % (float(dose_value))

                    let = plan.get_let_cube()
                    if let is not None:
                        let_value = let[pos[2], pos[1], pos[0]]
                        text += " / LET: %.1f kev/um" % (let_value)
            except IndexError as e:
                pass
            pub.sendMessage("statusbar.update", {"number": 2, "text": text})

    def normal_right_click_menu(self):
        menu = wx.Menu()
        voi_menu = wx.Menu()

        for voi in self.data.get_vois():
            id = wx.NewId()
            item = voi_menu.AppendCheckItem(id, voi.get_name())
            if voi.is_selected():
                item.Check()
            wx.EVT_MENU(self, id, self.menu_voi_selected)
        if voi_menu.GetMenuItemCount() > 0:
            menu.AppendSubMenu(voi_menu, "Vois")
        view_menu = wx.Menu()

        active_plan = self.active_plan
        if active_plan is not None:
            dose = active_plan.get_dose()
            dose_type_menu = wx.Menu()
            if dose is not None:
                id = wx.NewId()
                item = view_menu.AppendCheckItem(id, "View Dose")
                if self.plotutil.get_dose() is not None:
                    item.Check()
                wx.EVT_MENU(self, id, self.toogle_dose)

                id = wx.NewId()
                item = dose_type_menu.Append(id, "Color wash")
                wx.EVT_MENU(self, id, self.change_dose_to_colorwash)

                id = wx.NewId()
                item = dose_type_menu.Append(id, "Contour")
                wx.EVT_MENU(self, id, self.change_dose_to_contour)

                menu.AppendSubMenu(dose_type_menu, "Dose Visalization")
                if self.plotutil.get_dose_plot() == "contour":
                    dose_contour_menu = wx.Menu()
                    for level in self.dose_contour_levels:
                        id = wx.NewId()
                        item = dose_contour_menu.AppendCheckItem(id, "%d %%" % level)
                        for contour in self.plotutil.get_dose_contours():
                            if contour["doselevel"] == level:
                                item.Check()
                        wx.EVT_MENU(self, id, self.toogle_dose_contour)
                    menu.AppendSubMenu(dose_contour_menu, "Dose Contour levels")

            let = active_plan.get_let()

            if let is not None:
                id = wx.NewId()
                item = view_menu.AppendCheckItem(id, "View LET")
                if self.plotutil.get_let() is not None:
                    item.Check()
                wx.EVT_MENU(self, id, self.toogle_let)

            if view_menu.GetMenuItemCount() > 0:
                menu.AppendSubMenu(view_menu, "View")

            field_menu = wx.Menu()
            for field in active_plan.get_fields():
                id = wx.NewId()
                item = field_menu.AppendCheckItem(id, field.get_name())
                if field.is_selected():
                    item.Check()
                wx.EVT_MENU(self, id, self.menu_field_selected)
            if field_menu.GetMenuItemCount() > 0:
                menu.AppendSubMenu(field_menu, "Fields")

        jump_menu = wx.Menu()
        id = wx.NewId()
        item = jump_menu.Append(id, "First")
        wx.EVT_MENU(self, id, self.jump_to_first)

        id = wx.NewId()
        item = jump_menu.Append(id, "Middle")
        wx.EVT_MENU(self, id, self.jump_to_middle)

        id = wx.NewId()
        item = jump_menu.Append(id, "Last")
        wx.EVT_MENU(self, id, self.jump_to_last)

        menu.AppendSubMenu(jump_menu, "Jump To")
        return menu

    def right_click_dose(self):
        menu = wx.Menu()
        id = wx.NewId()
        item = menu.Append(id, "Reset")
        wx.EVT_MENU(menu, id, self.reset_dose_range)

        colormap_menu = wx.Menu()

        id = wx.NewId()
        colormap_menu.Append(id, "Continuous")
        wx.EVT_MENU(colormap_menu, id, self.set_colormap_dose)

        id = wx.NewId()
        colormap_menu.Append(id, "Discrete")
        wx.EVT_MENU(colormap_menu, id, self.set_colormap_dose)

        item = menu.AppendSubMenu(colormap_menu, "Colorscale")

        scale_menu = wx.Menu()

        id = wx.NewId()
        scale_menu.Append(id, "Auto")
        wx.EVT_MENU(scale_menu, id, self.set_dose_scale)

        if self.active_plan.get_dose().get_dose() > 0.0:
            id = wx.NewId()
            scale_menu.Append(id, "Absolute")
            wx.EVT_MENU(scale_menu, id, self.set_dose_scale)

        id = wx.NewId()
        scale_menu.Append(id, "Relative")
        wx.EVT_MENU(scale_menu, id, self.set_dose_scale)

        item = menu.AppendSubMenu(scale_menu, "Scale")

        return menu

    def right_click_contrast(self):
        menu = wx.Menu()
        id = wx.NewId()
        item = menu.Append(id, "Reset")
        wx.EVT_MENU(menu, id, self.reset_contrast)
        return menu

    def set_colormap_dose(self, evt):
        colormap = plt.get_cmap(None)
        name = evt.GetEventObject().GetLabel(evt.GetId())
        if name == "Discrete":
            colormap = cmap_discretize(colormap, 10)
        self.plotutil.set_colormap_dose(colormap)
        self.Draw()

    def set_dose_scale(self, evt):
        scale = {"auto": "auto", "absolute": "abs", "relative": "rel"}
        name = evt.GetEventObject().GetLabel(evt.GetId())
        self.plotutil.set_dose_axis(scale[name.lower()])
        self.Draw()

    def reset_dose_range(self, evt):
        self.plotutil.set_dose_min_max(0, 100)
        self.Draw()

    def reset_contrast(self, evt):
        contrast = [-100, 400]
        self.plotutil.set_contrast(contrast)
        self.Draw()

    def jump_to_first(self, evt):
        self.image_idx = 0
        self.Draw()

    def jump_to_middle(self, evt):
        self.image_idx = self.plotutil.get_images_count() / 2
        self.Draw()

    def jump_to_last(self, evt):
        self.image_idx = self.plotutil.get_images_count() - 1
        self.Draw()

    def toogle_dose_contour(self, evt):
        value = float(evt.GetEventObject().GetLabel(evt.GetId()).split()[0])
        if evt.IsChecked():
            self.plotutil.add_dose_contour({"doselevel": value, "color": "b"})
        else:
            for contour in self.plotutil.get_dose_contours():
                if contour["doselevel"] == value:
                    self.plotutil.remove_dose_contour(contour)
        self.Draw()

    def toogle_dose(self, evt):
        if self.plotutil.get_dose() is None:
            self.plotutil.set_dose(self.active_plan.get_dose().get_dosecube())
        else:
            self.plotutil.set_dose(None)
        self.Draw()

    def toogle_let(self, evt):
        if self.plotutil.get_let() is None:
            self.plotutil.set_let(self.active_plan.get_let())
        else:
            self.plotutil.set_let(None)
        self.Draw()

    def menu_voi_selected(self, evt):
        name = evt.GetEventObject().GetLabel(evt.GetId())
        name = name.replace("__", "_")
        voi = self.data.get_vois().get_voi_by_name(name)
        if not voi is None:
            voi.toogle_selected()

    def menu_field_selected(self, evt):
        name = evt.GetEventObject().GetLabel(evt.GetId())
        field = self.active_plan.get_fields().get_field_by_name(name)
        field.toogle_selected(self.active_plan)

    def change_dose_to_colorwash(self, evt):
        self.plotutil.set_dose_plot("colorwash")
        self.Draw()

    def change_dose_to_contour(self, evt):
        self.plotutil.set_dose_plot("contour")
        self.Draw()

    def on_size(self, evt):
        """Refresh the view when the size of the panel changes."""

        self.setSize()

    def on_mouse_wheel(self, evt):
        delta = evt.GetWheelDelta()
        rot = evt.GetWheelRotation()
        rot = rot / delta

        if evt.ControlDown():
            if (rot >= 1):
                self.zoom_in(None)
            elif (rot < 1):
                self.zoom_out(None)
            return
        n_images = self.data.get_images().get_voxelplan().dimz
        if n_images:
            if (rot >= 1):
                if (self.image_idx > 0):
                    self.image_idx -= 1
                    self.Draw()
            if (rot <= -1):
                if (self.image_idx < self.plotutil.get_images_count() - 1):
                    self.image_idx += 1
                    self.Draw()

    def on_key_down(self, evt):
        prevkey = [wx.WXK_UP, wx.WXK_PAGEUP]
        nextkey = [wx.WXK_DOWN, wx.WXK_PAGEDOWN]
        code = evt.GetKeyCode()
        if code in prevkey:
            if (self.image_idx > 0):
                self.image_idx -= 1
                self.Draw()
        elif code in nextkey:
            if (self.image_idx < self.plotutil.get_images_count() - 1):
                self.image_idx += 1
                self.Draw()

    def on_mouse_enter(self, evt):
        """Set a flag when the cursor enters the window."""
        self.mouse_in_window = True

    def on_mouse_leave(self, evt):
        """Set a flag when the cursor leaves the window."""

        self.mouse_in_window = False

    def setSize(self):
        size = self.parent.GetClientSize()
        size[1] = size[1] - 40
        size[0] = size[0] - 5
        pixels = tuple(size)
        self.canvas.SetSize(pixels)
        self.figure.set_size_inches(float(pixels[0]) / self.figure.get_dpi(),
                                    float(pixels[1]) / self.figure.get_dpi())
        self.Draw()

    def Draw(self):
        self.plotutil.plot(self.image_idx)

        self.figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        # self.figure.tight_layout(pad=0.0)

        if hasattr(self.plotutil, "dose_bar"):
            bar = self.plotutil.dose_bar
            bar.ax.yaxis.label.set_color('white')
            bar.ax.tick_params(axis='y', colors='white', labelsize=8)
        if hasattr(self.plotutil, "let_bar"):
            bar = self.plotutil.let_bar
            bar.ax.yaxis.label.set_color('white')
            bar.ax.yaxis.label.set_color('white')
            bar.ax.tick_params(axis='y', colors='white', labelsize=8)
        if hasattr(self.plotutil, "contrast_bar"):
            bar = self.plotutil.contrast_bar
            bar.ax.yaxis.label.set_color('white')
            bar.ax.yaxis.set_label_position('left')
            [t.set_color("white") for t in bar.ax.yaxis.get_ticklabels()]
            [t.set_size(8) for t in bar.ax.yaxis.get_ticklabels()]

            # bar.ax.tick_params(axis='y', colors='white',labelsize=8,labelleft=True,labelright=False)

        self.canvas.draw()
