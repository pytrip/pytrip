"""
    This file is part of PyTRiP.

    PyTRiP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyTRiP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyTRiP.  If not, see <http://www.gnu.org/licenses/>
"""
from pytrip.error import InputError

from pytrip.tripexecuter.pytripobj import pytripObj


class TripVoi(pytripObj):
    def __init__(self, voi):
        self.save_fields = ["name", "target", "max_dose_fraction", "oar", "dose", "hu_offset", "hu_value",
                            "dose_percent", "cube_value"]
        self._voi = voi
        self.name = voi.get_name()
        self.target = False
        self.max_dose_fraction = 100.0
        self.oar = False
        self.dose = 68
        self.hu_offset = None
        self.hu_value = None
        self.selected = False
        self.dvh = None
        self.lvh = None
        self.dose_percent = {}
        self.cube_value = -1

    def set_cube_value(self, value):
        self.cube_value = value

    def get_cube_value(self):
        return self.cube_value

    def get_dvh(self, dose):
        if dose is None:
            return None
        if self.dvh is None:
            (self.dvh, self.min_dose, self.max_dose, self.mean, area) = dose.calculate_dvh(self._voi.get_voi_data())
        return self.dvh

    def set_dose_percent(self, ion, dose_percent):
        if dose_percent == "":
            del self.dose_percent[ion]
        self.dose_percent[ion] = float(dose_percent)

    def get_dose_percent(self, ion):
        if ion in self.dose_percent:
            return self.dose_percent[ion]
        return None

    def get_all_dose_percent(self):
        return self.dose_percent

    def remove_dose_percent(self, ion):
        del self.dose_percent[ion]

    def get_hu_offset(self):
        return self.hu_offset

    def set_hu_offset(self, value):
        if len(value) is 0:
            self.hu_offset = None
            return
        try:
            value = float(value)
            self.hu_offset = value
        except Exception:
            raise InputError("HU Offset should be a number")

    def get_hu_value(self):
        return self.hu_value

    def set_hu_value(self, value):
        if len(value) is 0:
            self.hu_value = None
            return
        try:
            value = float(value)
            self.hu_value = value
        except Exception:
            raise InputError("HU Value should be a number")

    def get_dose(self):
        return self.dose

    def get_max_dose_fraction(self):
        return self.max_dose_fraction

    def get_lvh(self, let):
        if let is None:
            return None
        if self.lvh is None:
            (self.lvh, self.min_let, self.max_let, self.let, area) = let.calculate_lvh(self._voi.get_voi_data())
        return self.lvh

    def clean_cache(self):
        self.dvh = None

        # used for dvh or similar, plan selected

    def is_plan_selected(self):
        return self.selected

    def toogle_plan_selected(self, plan):
        self.selected = not self.selected

    def get_name(self):
        return self.name

    def get_voi(self):
        return self._voi

    def set_max_dose_fraction(self, max_dose_fraction):
        try:
            max_dose_fraction = float(max_dose_fraction)
            if max_dose_fraction < 0:
                raise Exception()
            self.max_dose_fraction = max_dose_fraction
        except Exception:
            raise InputError("Max dose fraction should be " "a number between 0 and 1")

    def set_dose(self, dose):
        try:
            dose = float(dose)
            if dose < 0:
                raise Exception()
            self.dose = dose
        except Exception:
            raise InputError("Dose should be a number larger or equal to 0")

    def toogle_target(self):
        if self.is_oar() is True:
            return False
        if self.target is True:
            self.target = False
        else:
            self.target = True
        return self.target

    def toogle_oar(self):
        if self.is_target() is True:
            return False
        if self.oar is True:
            self.oar = False
        else:
            self.oar = True
        return self.oar

    def is_target(self):
        return self.target

    # used used for 2d plot and similar, global selected
    def is_selected(self):
        return self._voi.is_selected()

    def toogle_selected(self):
        self._voi.toogle_selected()

    def is_oar(self):
        return self.oar
