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


class FieldCollection(pytripObj):
    def __init__(self, plan):
        self.save_fields = ["fields"]
        self.plan = plan
        self.fields = []

    def Init(self):
        pass

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        self.iter = 0
        return self

    def __getitem__(self, i):
        return self.fields[i]

    def next(self):
        if self.iter >= len(self.fields):
            raise StopIteration
        self.iter += 1
        return self.fields[self.iter - 1]

    def add_field(self, field):
        if field.get_name() == "":
            field.name = "Field %d" % (len(self.fields) + 1)
        self.fields.append(field)

    def remove_field(self, field):
        self.fields.remove(field)

    def get_field_by_id(self, field):
        return self.fields[field - 1]

    def get_fields(self):
        return self.fields

    def get_field_by_name(self, name):
        for field in self:
            if field.get_name() == name:
                return field
        return None

    def get_number_of_fields(self):
        return len(self.fields)

    def destroy(self):
        pass
