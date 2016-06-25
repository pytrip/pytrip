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
try:
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib._cntr as cntr
except:
    pass

import numpy as np
import time
import pytrip.res.point


class PlotUtil:
    def __init__(self):
        matplotlib.interactive(True)
        self.contrast = [-100, 400]
        self.vois = []
        self.plot_vois = True
        self.dose_plot = "colorwash"
        self.dosecontour_levels = []
        self.let_plot = "colorwash"
        self.dose_axis = "auto"
        self.colormap_dose = plt.get_cmap(None);
        self.colormap_let = plt.get_cmap(None)
        self.fields = []
        self.figure = plt
        self.draw_in_gui = False
        self.voi_plots = []
        self.plot_plan = "Transversal"
        self.draw_text = True
        self.factor = 10
        self.zoom = 100.0
        self.center = [50.0, 50.0]
        self.plan = None

    def add_dose_contour(self, dose_contour):
        self.dosecontour_levels.append(dose_contour)

    def get_dose_contours(self):
        return self.dosecontour_levels

    def remove_dose_contour(self, dose_contour):
        self.dosecontour_levels.remove(dose_contour)

    def set_plan(self, plan):
        self.plan = plan

    def get_colorbar(self):
        return self.dose_bar

    def set_zoom(self, zoom):
        if zoom < self.zoom:
            self.zoom = zoom
            offset = self.get_offset()
            size = self.get_size()
            width = float(size[0]) / self.zoom * 100.0
            height = float(size[1]) / self.zoom * 100.0
            center = [float(size[0]) * self.center[0] / 100, float(size[1]) * self.center[1] / 100]
            if offset[0] < 0:
                self.center[0] += (-offset[0]) * 100 / float(size[0])
            elif offset[0] + width > size[0]:
                self.center[0] -= (offset[0] + width - size[0]) * 100 / float(size[0])

            if offset[1] < 0:
                self.center[1] += (-offset[1]) * 100 / float(size[1])
            elif offset[1] + height > size[1]:
                self.center[1] -= (offset[1] + height - size[1]) * 100 / float(size[1])
        else:
            self.zoom = zoom

    def get_offset(self):
        size = self.get_size()
        width = float(size[0]) / self.zoom * 100.0
        height = float(size[1]) / self.zoom * 100.0
        center = [float(size[0]) * self.center[0] / 100, float(size[1]) * self.center[1] / 100]
        offset = [center[0] - width / 2, center[1] - height / 2]
        return offset

    def move_center(self, offset):
        if self.zoom == 100:
            return False

        size = self.get_size()
        width = float(size[0]) / self.zoom * 100.0
        height = float(size[1]) / self.zoom * 100.0
        scale_factor = self.fig_ct.get_figure().get_size_inches()[0] * self.fig_ct.get_figure().get_dpi() / width / 2

        center = [float(size[0]) * self.center[0] / 100, float(size[1]) * self.center[1] / 100]
        center[0] = center[0] - float(offset[0]) / scale_factor
        center[1] = center[1] + float(offset[1]) / scale_factor / self.aspect

        off = [center[0] - width / 2, center[1] - height / 2]

        if off[0] > 0 and off[0] + width < size[0]:
            self.center[0] = center[0] / float(size[0]) * 100
        if off[1] > 0 and off[1] + height < size[1]:
            self.center[1] = center[1] / float(size[1]) * 100
        return True

    def get_zoom(self):
        return self.zoom

    def get_images_count(self):
        if self.plot_plan == "Transversal":
            return len(self.ctx.cube)
        elif self.plot_plan == "Sagital":
            return len(self.ctx.cube[0, 0])
        elif self.plot_plan == "Coronal":
            return len(self.ctx.cube[0])

    def set_draw_in_gui(self, yes):
        self.draw_in_gui = yes

    def set_plot_plan(self, plot_plan):
        self.plot_plan = plot_plan
        self.clear()

    def clear(self):
        if hasattr(self, "fig_dose"):
            del self.fig_dose
        if hasattr(self, "dose_bar"):
            del self.dose_bar
        if hasattr(self, "contrast_bar"):
            del self.contrast_bar
        if hasattr(self, "let_bar"):
            del self.let_bar
        if hasattr(self, "fig_ct"):
            del self.fig_ct

    def set_colormap_dose(self, colormap):
        self.colormap_dose = colormap
        self.clear_dose_view()

    def set_figure(self, figure):
        self.figure = figure

    def set_dose_axis(self, dose_axis):
        self.dose_axis = dose_axis
        self.clear_dose_view()

    def set_colormap_let(self, colormap):
        self.colormap_let = colormap

    def set_ct(self, ctx):
        self.ctx = ctx

    def set_let(self, let):
        self.let = let
        if self.let is not None:
            self.max_let = np.amax(let.cube)
            self.min_let = 0

    def set_dose_plot(self, type):
        self.dose_plot = type
        self.clear_dose_view()

    def get_dose_plot(self):
        return self.dose_plot

    def set_dose(self, dos):
        self.dos = dos
        if self.dos is not None:
            self.max_dose = np.amax(dos.cube) / self.factor
            self.min_dose = 0
            if hasattr(self, "fig_dose"):
                self.fig_dose.set_clim(vmin=self.min_dose, vmax=self.max_dose)

    def set_dose_min_max(self, min_max):
        if min_max[0] >= 0 and min_max[0] < min_max[1]:
            self.min_dose = min_max[0]
            self.max_dose = min_max[1]
            if hasattr(self, "fig_dose"):
                self.fig_dose.set_clim(vmin=self.min_dose, vmax=self.max_dose)

    def set_let_min_max(self, min_max):
        if min_max[0] >= 0 and min_max[0] < min_max[1]:
            self.min_let = min_max[0]
            self.max_let = min_max[1]
            if hasattr(self, "fig_let"):
                self.fig_let.set_clim(vmin=self.min_let, vmax=self.max_let)

    def get_min_max_dose(self):
        return [self.min_dose, self.max_dose]

    def get_min_max_let(self):
        return [self.min_let, self.max_let]

    def add_voi(self, voi):
        self.vois.append(voi)

    def remove_voi(self, voi):
        self.vois.remove(voi)

    def set_contrast(self, contrast):
        if (contrast[0] >= contrast[1] or contrast[1] > 2000 or contrast[0] < -1000):
            return
        self.contrast = contrast
        if hasattr(self, "fig_ct"):
            self.fig_ct.set_clim(vmin=contrast[0], vmax=contrast[1])

    def pixel_to_pos(self, pixel):
        if self.plot_plan == "Transversal":
            pos = [pixel[0] * self.ctx.pixel_size, pixel[1] * self.ctx.pixel_size]
        elif self.plot_plan == "Sagital":
            pos = [(self.ctx.dimy - pixel[0]) * self.ctx.pixel_size,
                   (self.ctx.dimz - pixel[1]) * self.ctx.slice_distance]
        elif self.plot_plan == "Coronal":
            pos = [pixel[0] * self.ctx.pixel_size, (self.ctx.dimz - pixel[1]) * self.ctx.slice_distance]
        return [pixel, pos]

    def get_contrast(self):
        return self.contrast

    def get_dose(self):
        if hasattr(self, "dos"):
            return self.dos
        return None

    def get_size(self):
        if self.plot_plan == "Transversal":
            width = self.ctx.dimx
            height = self.ctx.dimy
        elif self.plot_plan == "Sagital":
            width = self.ctx.dimy
            height = self.ctx.dimz
        elif self.plot_plan == "Coronal":
            width = self.ctx.dimx
            height = self.ctx.dimz
        return [width, height]

    def get_let(self):
        if hasattr(self, "let"):
            return self.let
        return None

    def plot(self, idx):
        if self.plot_plan == "Transversal":
            ct_data = self.ctx.cube[idx]
            self.aspect = 1.0
        elif self.plot_plan == "Sagital":
            ct_data = self.ctx.cube[-1:0:-1, -1:0:-1, idx]
            self.aspect = self.ctx.slice_distance / self.ctx.pixel_size
        elif self.plot_plan == "Coronal":
            ct_data = self.ctx.cube[-1:0:-1, idx, -1:0:-1]
            self.aspect = self.ctx.slice_distance / self.ctx.pixel_size

        if not hasattr(self, "fig_ct"):
            self.fig_ct = self.figure.imshow(ct_data, cmap=plt.get_cmap("gray"), vmin=self.contrast[0],
                                             vmax=self.contrast[1], aspect=self.aspect)
        else:
            self.fig_ct.set_data(ct_data)

        if self.plot_plan == "Transversal":
            self.figure.axis([0, self.ctx.dimx, self.ctx.dimy, 0])
        elif self.plot_plan == "Sagital":
            self.figure.axis([0, self.ctx.dimy, self.ctx.dimz, 0])
        elif self.plot_plan == "Coronal":
            self.figure.axis([0, self.ctx.dimx, self.ctx.dimz, 0])
        if self.draw_in_gui:
            self.figure.axes.get_xaxis().set_visible(False)
            self.figure.axes.get_yaxis().set_visible(False)
            if not hasattr(self, "contrast_bar"):
                cax = self.figure.figure.add_axes([0.1, 0.1, 0.03, 0.8])
                self.contrast_bar = self.figure.figure.colorbar(self.fig_ct, cax=cax)
        # ~
        self.clean_plot()
        self.plot_dose(idx)
        self.plot_let(idx)

        if self.plot_vois:
            for i, voi in enumerate(self.vois):
                plot = False
                data = []
                if self.plot_plan == "Transversal":
                    slice = voi.get_slice_at_pos(self.ctx.slice_to_z(idx))
                    if slice is None:
                        continue
                    for contour in slice.contour:
                        data.append(np.array(contour.contour))
                        plot = True
                elif self.plot_plan == "Sagital":
                    slice = voi.get_2d_slice(voi.sagital, idx * self.ctx.pixel_size)
                    if not slice is None:
                        data.append(np.array(slice.contour[0].contour))
                        plot = True
                elif self.plot_plan == "Coronal":
                    slice = voi.get_2d_slice(voi.coronal, idx * self.ctx.pixel_size)
                    if not slice is None:
                        data.append(np.array(slice.contour[0].contour))
                        plot = True
                data = self.points_to_plane(data)
                if plot:
                    for d in data:
                        self.figure.plot(d[:, 0], d[:, 1], color=(np.array(voi.get_color()) / 255.0))
        # set zoom
        size = self.get_size()
        width = float(size[0]) / self.zoom * 100.0
        height = float(size[1]) / self.zoom * 100.0
        offset = self.get_offset()
        self.figure.axes.set_xlim(offset[0], offset[0] + width)
        self.figure.axes.set_ylim(offset[1] + height, offset[1])

        self.plot_text(idx)
        self.plot_fields(idx)
        if not self.draw_in_gui:
            self.figure.show()

    def clean_plot(self):
        while len(self.figure.lines) > 0:
            self.figure.lines.pop(0)
        while len(self.figure.texts) > 0:
            self.figure.texts.pop(0)

    def show(self):
        self.figure.show()

    def points_to_plane(self, point):
        d = point
        for data in d:
            if self.plot_plan == "Transversal":
                data[:, 0] /= self.ctx.pixel_size
                data[:, 1] /= self.ctx.pixel_size
            elif self.plot_plan == "Sagital":
                data[:, 0] = (-data[:, 1] + self.ctx.pixel_size * self.ctx.dimx) / self.ctx.pixel_size
                data[:, 1] = (-data[:, 2] + self.ctx.slice_distance * self.ctx.dimz) / self.ctx.slice_distance
            elif self.plot_plan == "Coronal":
                data[:, 0] = (-data[:, 0] + self.ctx.pixel_size * self.ctx.dimy) / self.ctx.pixel_size
                data[:, 1] = (-data[:, 2] + self.ctx.slice_distance * self.ctx.dimz) / self.ctx.slice_distance
        return d

    def plot_fields(self, idx):
        if self.plan is None:
            return
        targets = []
        for v in self.plan.get_vois():
            if v.is_target():
                targets.append(v)
        if len(targets) is 0 or len(targets) > 1:
            return

        target = targets[0].get_voi().get_voi_data()
        center = target.calculate_center()
        data = None
        if self.plot_plan == "Transversal":
            slice = target.get_slice_at_pos(self.ctx.slice_to_z(idx))
            if not slice is None:
                for contour in slice.contour:
                    data = np.array(contour.contour)
            vec = np.array([0, 0, 1])
        elif self.plot_plan == "Sagital":
            slice = target.get_2d_slice(target.sagital, idx * self.ctx.pixel_size)
            if not slice is None:
                for contour in slice.contour:
                    data = np.array(contour.contour)
            vec = np.array([0, 1, 0])
        elif self.plot_plan == "Coronal":
            slice = target.get_2d_slice(target.coronal, idx * self.ctx.pixel_size)
            if not slice is None:
                for contour in slice.contour:
                    data = np.array(contour.contour)
            vec = np.array([1, 0, 0])

        if data is None:
            return
        for f in self.plan.get_fields():
            if not f.is_selected():
                continue

            gantry = f.get_gantry()
            couch = f.get_couch()
            field_vec = -res.point.get_basis_from_angles(gantry, couch)[0]

            field_vec = field_vec - np.dot(field_vec, vec) * vec

            if abs(np.linalg.norm(field_vec)) < 0.01:
                continue

            field_vec /= np.linalg.norm(field_vec)
            cross_vec = np.cross(field_vec, vec)
            proj = np.dot(data - slice.calculate_center()[0], np.transpose(cross_vec))
            idx_min = np.where(min(proj) == proj)[0][0]
            idx_max = np.where(max(proj) == proj)[0][0]
            point_min = data[idx_min]
            point_max = data[idx_max]

            steps = int(self.ctx.pixel_size * self.ctx.dimx / 2)
            f1 = np.array([point_min + x * field_vec for x in range(steps)])
            f2 = np.array([point_max + x * field_vec for x in range(steps)])
            plot_data = []

            plot_data.append(f1)
            plot_data.append(f2)
            plot_data = self.points_to_plane(plot_data)

            color = np.array([124, 252, 0]) / 255.0
            self.figure.plot(plot_data[0][:, 0], plot_data[0][:, 1], color=color)
            self.figure.plot(plot_data[1][:, 0], plot_data[1][:, 1], color=color)

    def clear_dose_view(self):
        if hasattr(self, "fig_dose"):
            self.fig_dose.set_visible(False)
            for ax in self.figure.figure.axes:
                if ax is self.dose_bar.ax:
                    self.figure.figure.delaxes(ax)
            del self.fig_dose
        if hasattr(self, "dose_bar"):
            for ax in self.figure.figure.axes:
                if ax is self.dose_bar.ax:
                    self.figure.figure.delaxes(ax)
            del self.dose_bar

    def plot_dose(self, idx):
        if not hasattr(self, "dos"):
            return
        if self.dos is None:
            self.clear_dose_view()
            del self.dos
            return
        if self.plot_plan == "Transversal":
            dos_data = self.dos.cube[idx]
        elif self.plot_plan == "Sagital":
            dos_data = self.dos.cube[-1:0:-1, -1:0:-1, idx]
        elif self.plot_plan == "Coronal":
            dos_data = self.dos.cube[-1:0:-1, idx, -1:0:-1]

        if self.dose_plot == "colorwash":
            if self.dos.target_dose <= 0:
                scale = "rel"
            elif self.dose_axis == "auto" and self.dos.target_dose is not 0.0:
                scale = "abs"
            else:
                scale = self.dose_axis
            if scale == "abs":
                self.factor = 1000 / self.dos.target_dose
            if scale == "rel":
                self.factor = 10
            if hasattr(self, "scale") and not self.scale == scale:
                self.max_dose = np.amax(self.dos.cube) / self.factor
                self.clear_dose_view()
            elif not hasattr(self, "scale"):
                self.max_dose = np.amax(self.dos.cube) / self.factor

            cmap = self.colormap_dose
            cmap._init()
            cmap._lut[:, -1] = 0.7
            cmap._lut[0, -1] = 0.0

            plot_data = dos_data / float(self.factor)
            plot_data[plot_data <= self.min_dose] = self.min_dose
            if not hasattr(self, "fig_dose") or not self.scale == scale:
                self.fig_dose = self.figure.imshow(plot_data, cmap=cmap, vmax=(self.max_dose), aspect=self.aspect)
                if not self.draw_in_gui:
                    bar = self.figure.colorbar()
                else:
                    if not hasattr(self, "dose_bar") and not hasattr(self, "let_bar"):
                        cax = self.figure.figure.add_axes([0.9, 0.1, 0.03, 0.8])
                        self.dose_bar = self.figure.figure.colorbar(self.fig_dose, cax=cax)
                if scale == "abs":
                    self.dose_bar.set_label("Dose (Gy)")
                else:
                    self.dose_bar.set_label("Dose (%)")
            else:
                self.fig_dose.set_data(plot_data)
            self.scale = scale
        elif self.dose_plot == "contour":
            x, y = np.meshgrid(np.arange(len(dos_data[0])), np.arange(len(dos_data)))
            isodose_obj = cntr.Cntr(x, y, dos_data)
            for con in self.dosecontour_levels:
                contour = isodose_obj.trace(con["doselevel"] * 10)
                if len(contour) > 0:
                    self.figure.plot(contour[0][:, 0], contour[0][:, 1], con["color"])

    def plot_text(self, idx):
        size = self.get_size()
        offset = self.get_offset()
        width = size[0]
        height = size[1]
        if self.plot_plan == "Transversal":
            slices = self.ctx.dimz
            slice_dist = self.ctx.slice_distance
        elif self.plot_plan == "Sagital":
            slices = self.ctx.dimy
            slice_dist = self.ctx.pixel_size
        elif self.plot_plan == "Coronal":
            slices = self.ctx.dimx
            slice_dist = self.ctx.pixel_size
        self.figure.text(offset[0], offset[1] + 3.0 / self.zoom * 100,
                         "Slice #: %d/%d\nSlice Position: %.1f mm" % (idx, slices, idx * slice_dist), color="white",
                         va="top", fontsize=8)
        self.figure.text(offset[0] + width / self.zoom * 100, offset[1] + 3.0 / self.zoom * 100,
                         "W / L: %d / %d" % (self.contrast[1], self.contrast[0]), ha="right", color="white", va="top",
                         fontsize=8)

        self.figure.text(offset[0], offset[1] + (height - 5) / self.zoom * 100, self.plot_plan, color="white",
                         va="bottom", fontsize=8)

    def plot_let(self, idx):
        if not hasattr(self, "let"):
            return
        if self.let is None:
            if hasattr(self, "fig_let"):
                self.fig_let.set_visible(False)
                del self.fig_let
                del self.let
            if hasattr(self, "let_bar"):
                for ax in self.figure.figure.axes:
                    if ax is self.let_bar.ax:
                        self.figure.figure.delaxes(ax)
                del self.let_bar
            return

        if self.let_plot == "colorwash":
            cmap = self.colormap_dose
            cmap._init()
            cmap._lut[:, -1] = 0.7
            cmap._lut[0, -1] = 0.0
            if self.plot_plan == "Transversal":
                let_data = self.let.cube[idx]
            elif self.plot_plan == "Sagital":
                let_data = self.let.cube[-1:0:-1, idx, -1:0:-1]
            elif self.plot_plan == "Coronal":
                let_data = self.let.cube[-1:0:-1, -1:0:-1, idx]

            let_data[let_data <= self.min_let] = self.min_let
            if not hasattr(self, "fig_let"):
                self.fig_let = self.figure.imshow(let_data, cmap=cmap, vmax=(self.max_let), aspect=self.aspect)
                if not self.draw_in_gui:
                    bar = self.figure.colorbar()
                else:
                    if not hasattr(self, "let_bar") and not hasattr(self, "dose_bar"):
                        cax = self.figure.figure.add_axes([0.9, 0.1, 0.03, 0.8])
                        self.let_bar = self.figure.figure.colorbar(self.fig_let, cax=cax)
                        self.let_bar.set_label("LET (keV/um)")
            else:
                self.fig_let.set_data(let_data)
