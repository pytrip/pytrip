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
from pytrip.tripexecuter.pytripobj import pytripObj
from pytrip.tripexecuter.tripvoi import TripVoi
from pytrip.util import get_class_name


class VoiCollection(pytripObj):
    def __init__(self, parent):
        self.save_fields = ["vois"]
        self.parent = parent
        self.vois = []

    def Init(self):
        pass

    def __iter__(self):
        self.iter = 0
        return self

    def __len__(self):
        return len(self.vois)

    def __getitem__(self, i):
        return self.vois[i]

    def next(self):
        if self.iter >= len(self.vois):
            raise StopIteration
        self.iter += 1
        return self.vois[self.iter - 1]

    def get_vois(self):
        return self.vois

    def get_voi_by_name(self, name):
        for voi in self:
            if voi.get_name() == name:
                return voi
        return None

    def move_voi(self, voi, step):
        for i, v in enumerate(self.vois):
            if v is voi:
                break
        i2 = i + step
        if i < 0 or i >= len(self.vois):
            return False
        else:
            tmp = self.vois[i2]
            self.vois[i2] = self.vois[i]
            self.vois[i] = tmp
            return True

    def get_target_name(self):
        return ""

    def add_voi(self, voi):
        if voi in self.vois:
            return False
        if get_class_name(self.parent) == "TripPlan":
            if TripVoi not in voi.__class__.__bases__:
                voi = TripVoi(voi)
            for v in self.vois:
                if v._voi is voi._voi:
                    return False
        self.vois.append(voi)
        return True

    def delete_voi(self, voi):
        self.vois.remove(voi)
