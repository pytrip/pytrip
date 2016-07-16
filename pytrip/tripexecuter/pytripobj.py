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


class pytripObj(object):
    def save(self):
        data = {}
        if hasattr(self, "save_fields"):
            for field in self.save_fields:
                item = getattr(self, field)
                if hasattr(item, "save"):
                    data[field] = item.save()
                elif type(item) is list:
                    if len(item) > 0:
                        data[field] = []
                        for i in item:
                            if hasattr(i, "save"):
                                data[field].append(i.save())
                            else:
                                data[field].append(i)
                elif type(item) is dict:
                    if len(item) > 0:
                        data[field] = {}
                        for i in item.keys():
                            if hasattr(item[i], "save"):
                                data[field][i] = item[i].save()
                            else:
                                data[field] = item
                else:
                    data[field] = item
        return data
