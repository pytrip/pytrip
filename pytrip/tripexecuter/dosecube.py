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


class DoseCube(object):
    def __init__(self, dosecube, type):
        self.dosecube = dosecube
        self.type = type

    def get_dosecube(self):
        return self.dosecube

    def get_type(self):
        return self.type

    def set_type(self, type):
        self.type = type

    def set_dose(self, value):
        self.dosecube.target_dose = float(value)

    def get_dose(self):
        return self.dosecube.target_dose

    def calculate_dvh(self, voi):
        dvh, min_dose, max_dose, mean, area = self.dosecube.calculate_dvh(voi)
        return dvh, min_dose, max_dose, mean, area
