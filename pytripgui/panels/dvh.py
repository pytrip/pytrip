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
import pdb
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

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
import matplotlib
import numpy as np
import functools


class LinePlotPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.parent = parent

    def Init(self, data):
        self.data = data
        self.figure = Figure(None, 100)
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.subplot = self.figure.add_subplot(111)
        self.subplot.grid(True)

        self.subplot.set_xlabel("%s (%s)" % (self.quantity, self.unit), fontsize=10)
        self.subplot.set_ylabel("Vol (%)", fontsize=10)
        self.set_color()
        self.bind()
        pub.subscribe(self.on_patient_updated, "patient.loaded")
        pub.sendMessage("patient.request")

    def get_figure(self):
        return self.figure

    def __del__(self):
        pub.unsubscribe(self.on_patient_updated)

    def on_voi_selection_change(self, msg):
        if msg.data["plan"] is self.plan:
            self.redraw()

    def on_patient_updated(self, msg):
        self.data = msg.data

    def set_size(self):
        size = self.parent.GetClientSize()
        size[1] = size[1] - 40
        size[0] = size[0] - 5
        pixels = tuple(size)
        self.canvas.SetSize(pixels)
        self.figure.set_size_inches(float(pixels[0]) / self.figure.get_dpi(),
                                    float(pixels[1]) / self.figure.get_dpi())
        self.draw()

    def bind(self):
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.canvas.Bind(wx.EVT_RIGHT_DOWN, self.on_mouse_right_click)
        self.canvas.Bind(wx.EVT_KEY_DOWN, self.on_key_down)

    def on_key_down(self, evt):
        if hasattr(self, "nearest_point"):
            prevkey = [wx.WXK_LEFT, wx.WXK_PAGEUP]
            nextkey = [wx.WXK_RIGHT, wx.WXK_PAGEDOWN]
            code = evt.GetKeyCode()
            if code in prevkey:
                if self.nearest_point["idx"] > 0:
                    self.nearest_point["idx"] -= 1
            elif code in nextkey:
                data = getattr(self.nearest_point["voi"], self.data_method)(
                    getattr(self.nearest_point["plan"], self.cube_method)())
                if self.nearest_point["idx"] < np.where(data <= 0)[0][0]:
                    self.nearest_point["idx"] += 1
            self.draw_nearest_text()
            self.draw()

    def on_mouse_right_click(self, evt):
        menu = wx.Menu()
        plan_menu = wx.Menu()

        for k, plan in enumerate(self.data.get_plans()):
            voi_menu = wx.Menu()
            for voi in plan.get_vois():
                id = wx.NewId()
                item = voi_menu.AppendCheckItem(id, voi.get_name())

                if voi.is_plan_selected():
                    item.Check()
                f = functools.partial(self.voi_selected, plan)
                wx.EVT_MENU(self, id, f)
            plan_menu.AppendSubMenu(voi_menu, "Plan %d" % (k + 1))
        menu.AppendSubMenu(plan_menu, "Plans")
        self.PopupMenu(menu, evt.GetPosition())
        menu.Destroy()

    def set_color(self):
        """Set figure and canvas colours to be the same."""
        rgbtuple = wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE).Get()
        clr = [c / 255. for c in rgbtuple]
        self.figure.set_facecolor(clr)
        self.figure.set_edgecolor(clr)
        self.canvas.SetBackgroundColour(wx.Colour(*rgbtuple))

    def voi_selected(self, plan, evt):
        name = evt.GetEventObject().GetLabel(evt.GetId()).replace("__", "_")
        for voi in plan.get_vois():
            if voi.get_name() == name:
                voi.toogle_plan_selected(plan)
        self.redraw()

    def on_mouse_click(self, evt):
        if evt.button is 1:
            if not evt.inaxes:
                return
            if self.find_nearest_point([evt.xdata, evt.ydata]):
                self.draw_nearest_text()
            self.draw()

    def draw_nearest_text(self):
        while len(self.subplot.texts) > 0:
            self.subplot.texts.pop(0)
        font = matplotlib.font_manager.FontProperties(size=8)
        idx = self.nearest_point["idx"]
        plan = self.nearest_point["plan"]
        data = getattr(self.nearest_point["voi"], self.data_method)(getattr(plan, self.cube_method)())
        point = [data[idx], idx / 10.0]
        text = "Vol: " + unicode("%.2f" % (point[0] * 100)) + "%%\n%s: " % self.quantity + unicode(
            "%.2f" % (point[1])) + "%s" % self.unit
        if data[idx] < max(data) * 0.20:
            va = 'top'
            y_offset = 0.1
        else:
            va = 'bottom'
            y_offset = -0.1
        x_offset = -15
        ha = 'left'
        self.subplot.annotate(text, xy=(point[1], point[0] * 100), va=va, ha=ha,
                              xytext=(point[1] + x_offset, (point[0] + y_offset) * 100),
                              arrowprops=dict(arrowstyle="->", facecolor='black'), fontproperties=font)

    def find_nearest_point(self, point):
        g_min = None
        for plan in self.data.get_plans():
            for voi in plan.get_vois():
                if voi.is_plan_selected():
                    data = getattr(voi, self.data_method)(getattr(plan, self.cube_method)())
                    n = np.where(data <= 0.0)[0][0]
                    x = np.linspace(0, n / 10, n)
                    y = data[0:n] * 100
                    dist = (x - point[0]) ** 2 + ((y - point[1])) ** 2
                    idx = np.where(dist == min(dist))[0][0]
                    if g_min is None or dist[idx] < g_min:
                        g_min = dist[idx]
                        self.nearest_point = {"voi": voi, "idx": idx, "plan": plan}
        if g_min is not None:
            return True
        return False

    def redraw(self):
        if hasattr(self, "nearest_point"):
            del self.nearest_point
        while len(self.subplot.lines) > 0:
            self.subplot.lines.pop(0)
        while len(self.subplot.texts) > 0:
            self.subplot.texts.pop(0)
        self.subplot.legend_ = None
        markertypes = ['solid', 'dashed', 'dashdot', 'dotted']
        for i, plan in enumerate(self.data.get_plans()):
            for voi in plan.get_vois():
                if voi.is_plan_selected():
                    data = getattr(voi, self.data_method)(getattr(plan, self.cube_method)())
                    if data is None:
                        continue
                    n = np.where(data <= 0.0)[0][0]
                    self.subplot.plot(np.linspace(0, n / 10, n), data[0:n] * 100,
                                      label="%s: %s" % (plan.get_name(), voi.get_name()),
                                      color=(np.array(voi.get_voi().get_color()) / 255.0), linestyle=markertypes[i % 4])
            handles, labels = self.subplot.get_legend_handles_labels()

        self.subplot.legend(handles[::-1], labels[::-1], fancybox=True, prop={'size': 8})
        self.draw()

    # ~ Could be dosecube or LETCube
    def set_cube_method(self, cube):
        self.cube_method = cube

    def set_data_method(self, method):
        self.data_method = method

    def set_unit(self, unit):
        self.unit = unit

    def set_quantity(self, quantity):
        self.quantity = quantity

    def draw(self):
        self.canvas.draw()

    def on_size(self, evt):
        self.set_size()


class DVHPanel(LinePlotPanel):
    def __init__(self, parent):
        LinePlotPanel.__init__(self, parent)
        self.set_cube_method("get_dose")
        self.set_data_method("get_dvh")
        self.set_quantity("Dose")
        self.set_unit("%")
        pub.subscribe(self.dose_changed, "plan.dose.active_changed")

    def dose_changed(self, msg):
        self.redraw()

    def get_title(self):
        return "DVH"


class LVHPanel(LinePlotPanel):
    def __init__(self, parent):
        LinePlotPanel.__init__(self, parent)
        self.set_cube_method("get_let")
        self.set_data_method("get_lvh")
        self.set_quantity("LET")
        self.set_unit("keV/um")

    def get_title(self):
        return "LVH";
