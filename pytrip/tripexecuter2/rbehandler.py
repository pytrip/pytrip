#
#    Copyright (C) 2010-2017 PyTRiP98 Developers.
#
#    This file is part of PyTRiP98.
#
#    PyTRiP98 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyTRiP98 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyTRiP98.  If not, see <http://www.gnu.org/licenses/>.
#
"""
TODO: documentation here.
"""
import os
import csv


class RBEHandler:
    """ Class for managing .rbe files.
    """
    def __init__(self, datafile=""):
        self.datafile = datafile

    def get_rbe_by_name(self, name):
        for rbe in self.get_rbe_list():
            if rbe.get_name() == name:
                return rbe

    def get_rbe_list(self):
        if not hasattr(self, "rbe"):
            self.load_rbe()
        if hasattr(self, "rbe"):
            return self.rbe
        return []

    def load_rbe(self, i=0):
        """ Loads a single RBE file and stroed it into self.
        If file does not exist, then assume a directory with multiple RBE files, which
        then are loaded individually.
        TODO: refactor me, please.
        """
        if os.path.exists(self.datafile):
            with open(self.datafile, "r") as fp:
                reader = csv.reader(fp, delimiter='\t')
                self.rbe = [RBE(x[0], x[1]) for x in reader]  # RBE class holds name and path to RBE file.
                # TODO: above line looks broken, where does the data go?
        else:
            if i is 0:
                self._load_rbe_dir()
                self.load_rbe(1)

    def _load_rbe_dir(self):
        """ Used by load_rbe() if not a single file is present, to discover next files.
        """
        if not hasattr(self, "rbe_dir") or self.rbe_dir is None:
            return
        path = os.path.expandvars(self.rbe_dir)
        folder = os.listdir(path)
        with open(self.datafile, "w+") as fp_out:
            for item in folder:
                if os.path.splitext(item)[1] == ".rbe":
                    stop = False
                    with open(os.path.join(path, item), "r") as fp:
                        while not stop:
                            line = fp.readline()
                            if line.find("!celltype") > -1:
                                fp_out.write("%s\t%s\n" % (line.split()[1], os.path.join(path, item)))
                                stop = True
                            if not line:
                                stop = True


class RBE:
    def __init__(self, name="", path=""):
        """
        TODO: where does the data go?
        """
        self.name = name
        self.path = path
