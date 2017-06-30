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
Module with auxiliary functions (mostly internal use).
"""

import os
import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_class_name(item):
    """
    :returns: name of class of 'item' object.
    """
    return item.__class__.__name__


class TRiP98FilePath(object):
    """
    Helper class which deals with filename discovery
    TRiP98 files (cubes) are named according to one of the following pattern:

    A) STEM pattern

    STEM + SUFFIX + . + hed  (header file)
    STEM + SUFFIX + . + EXTENSION  (data file)

    STEM is a core part of the path provided by user (i.e. patient name), it can end with DOT (.)
    SUFFIX is used to distinguish between quantities saved in a cube (i.e. PHYS, BIO, RBE, SVV, MLET, DOSEMLET

    B) NAME pattern

    Here suffix is omitted and we have no information about quantity saved in a cube.

    NAME + . + hed (header file)
    NAME + . + EXTENSION (data file)

    Each of the files can also be gziped packed and end with .gz

    See http://bio.gsi.de/DOCS/TRiP98/PRO/DOCS/trip98cmddose.html

    In TRiP98 if user wants to write a cube to the file, an argument to the write command has to be provided.
    If the argument ends with proper data cube extension (i.e. .dos or .DOS for dose cube) then NAME pattern is used.
    Otherwise the argument is treated as STEM.
    """
    def __init__(self, name, cube_type):
        self.name = name
        self.cube_type = cube_type
        self.data_file_extension = self.cube_type.data_file_extension
        self.header_file_extension = self.cube_type.header_file_extension
        self.allowed_suffix = self.cube_type.allowed_suffix

    def is_valid_cube_type(self):
        no_suffix = self.suffix is None
        return no_suffix or (self.suffix in self.allowed_suffix)

    def is_valid_header_path(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("file.txt", pt.DosCube).is_valid_header_path()
        False
        >>> TRiP98FilePath("file.txt.hed", pt.DosCube).is_valid_header_path()
        True
        >>> TRiP98FilePath("file.HED", pt.DosCube).is_valid_header_path()
        True
        >>> TRiP98FilePath("file.DOS", pt.DosCube).is_valid_header_path()
        False
        >>> TRiP98FilePath("file.hed", pt.Cube).is_valid_header_path()
        True
        >>> TRiP98FilePath("file.hed", pt.LETCube).is_valid_header_path()
        True

        :return:
        """

        if not self.header_file_extension:
            return False

        # check lower and uppercase extensions
        return self._ungzipped_filename.endswith(
            (self.header_file_extension.lower(),
             self.header_file_extension.upper()))

    def _has_valid_datafile_extension(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("file.txt", pt.DosCube)._has_valid_datafile_extension()
        False
        >>> TRiP98FilePath("file.txt.hed", pt.DosCube)._has_valid_datafile_extension()
        False
        >>> TRiP98FilePath("file.dos", pt.DosCube)._has_valid_datafile_extension()
        True
        >>> TRiP98FilePath("file.DOS", pt.DosCube)._has_valid_datafile_extension()
        True
        >>> TRiP98FilePath("file.dos", pt.Cube)._has_valid_datafile_extension()
        False
        >>> TRiP98FilePath("file.dos", pt.LETCube)._has_valid_datafile_extension()
        True
        >>> TRiP98FilePath("file.dosemlet.dos", pt.DosCube)._has_valid_datafile_extension()
        True
        >>> TRiP98FilePath("file.dosemlet.dos", pt.LETCube)._has_valid_datafile_extension()
        True

        :return:
        """

        # check if corresponding cube class has information about data file extension
        if not self.data_file_extension:
            return False

        # check lower and uppercase extensions
        correct_extension = self._ungzipped_filename.endswith(
            (self.data_file_extension.lower(),
             self.data_file_extension.upper()))

        return correct_extension

    def is_valid_datafile_path(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("file.txt", pt.DosCube).is_valid_datafile_path()
        False
        >>> TRiP98FilePath("file.txt.hed", pt.DosCube).is_valid_datafile_path()
        False
        >>> TRiP98FilePath("file.dos", pt.DosCube).is_valid_datafile_path()
        True
        >>> TRiP98FilePath("file.DOS", pt.DosCube).is_valid_datafile_path()
        True
        >>> TRiP98FilePath("file.dos", pt.Cube).is_valid_datafile_path()
        False
        >>> TRiP98FilePath("file.dos", pt.LETCube).is_valid_datafile_path()
        True
        >>> TRiP98FilePath("file.dosemlet.dos", pt.DosCube).is_valid_datafile_path()
        True
        >>> TRiP98FilePath("file.dosemlet.dos", pt.LETCube).is_valid_datafile_path()
        True

        :return:
        """

        # check lower and uppercase extensions
        correct_extension = self._has_valid_datafile_extension()

        # check suffix if present
        suffix = self.suffix
        if suffix is None:
            compatible_suffix = True
        else:
            compatible_suffix = suffix in self.allowed_suffix

        return correct_extension and compatible_suffix

    def _is_gzipped(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("file.txt", pt.Cube)._is_gzipped()
        False
        >>> TRiP98FilePath("file.txt.gz", pt.Cube)._is_gzipped()
        True
        >>> TRiP98FilePath("file.txt.GZ", pt.Cube)._is_gzipped()
        True
        >>> TRiP98FilePath("file.txt.gZ", pt.Cube)._is_gzipped()
        False

        :return:
        """
        return self.name.endswith(('.gz', '.GZ'))

    @property
    def _ungzipped_filename(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("file.txt", pt.Cube)._ungzipped_filename
        'file.txt'
        >>> TRiP98FilePath("file.txt.gz", pt.Cube)._ungzipped_filename
        'file.txt'
        >>> TRiP98FilePath("file.txt.GZ", pt.Cube)._ungzipped_filename
        'file.txt'
        >>> TRiP98FilePath("file.txt.gZ", pt.Cube)._ungzipped_filename
        'file.txt.gZ'

        :return:
        """
        if not self._is_gzipped():
            return self.name
        else:
            return self.name[:-3]

    @property
    def _filename_without_extension(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("file.txt", pt.Cube)._filename_without_extension
        'file.txt'
        >>> TRiP98FilePath("file.txt.gz", pt.Cube)._filename_without_extension
        'file.txt'
        >>> TRiP98FilePath("file.txt.dos", pt.DosCube)._filename_without_extension
        'file.txt'
        >>> TRiP98FilePath("file.hed", pt.DosCube)._filename_without_extension
        'file'
        >>> TRiP98FilePath("file.txt.dos", pt.CtxCube)._filename_without_extension
        'file.txt.dos'
        """
        if self.is_valid_header_path():
            return self._ungzipped_filename[:-len(self.header_file_extension)]
        elif self._has_valid_datafile_extension():
            return self._ungzipped_filename[:-len(self.data_file_extension)]
        else:
            return self._ungzipped_filename

    @property
    def suffix(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("file.txt", pt.Cube).suffix
        >>> TRiP98FilePath("file.txt", pt.DosCube).suffix
        >>> TRiP98FilePath("filephys.txt", pt.DosCube).suffix
        >>> TRiP98FilePath("filephys.dos", pt.DosCube).suffix
        'phys'
        >>> TRiP98FilePath("filephys.hed", pt.DosCube).suffix
        'phys'
        >>> TRiP98FilePath("filephys.hed", pt.LETCube).suffix
        >>> TRiP98FilePath("filemlet.hed", pt.LETCube).suffix
        'mlet'
        >>> TRiP98FilePath("file.mlet.hed", pt.LETCube).suffix
        'mlet'

        :return:
        """
        _filename_without_extension = self._filename_without_extension
        if not self.allowed_suffix:
            return None
        for suffix in self.allowed_suffix:
            if _filename_without_extension.endswith((suffix.lower(), suffix.upper())):
                return _filename_without_extension[-len(suffix):]
        return None

    @property
    def stem(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("file.txt", pt.Cube).stem
        >>> TRiP98FilePath("file.txt", pt.DosCube).stem
        >>> TRiP98FilePath("filephys.txt", pt.DosCube).stem
        >>> TRiP98FilePath("filephys.dos", pt.DosCube).stem
        'file'
        >>> TRiP98FilePath("filephys.hed", pt.DosCube).stem
        'file'
        >>> TRiP98FilePath("filephys.hed", pt.LETCube).stem
        >>> TRiP98FilePath("filemlet.hed", pt.LETCube).stem
        'file'
        >>> TRiP98FilePath("file.mlet.hed", pt.LETCube).stem
        'file.'

        :return:
        """
        if self._is_stem_pattern():
            return self._filename_without_extension[:-len(self.suffix)]
        else:
            return None

    def _is_stem_pattern(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("file.txt", pt.Cube)._is_stem_pattern()
        False
        >>> TRiP98FilePath("file.txt", pt.DosCube)._is_stem_pattern()
        False
        >>> TRiP98FilePath("filephys.txt", pt.DosCube)._is_stem_pattern()
        False
        >>> TRiP98FilePath("filephys.dos", pt.DosCube)._is_stem_pattern()
        True
        >>> TRiP98FilePath("filephys.hed", pt.DosCube)._is_stem_pattern()
        True
        >>> TRiP98FilePath("filephys.hed", pt.LETCube)._is_stem_pattern()
        False
        >>> TRiP98FilePath("filemlet.hed", pt.LETCube)._is_stem_pattern()
        True
        >>> TRiP98FilePath("file.mlet.hed", pt.LETCube)._is_stem_pattern()
        True

        :return:
        """
        return bool(self.suffix)

    def _is_name_pattern(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("file.txt", pt.Cube)._is_name_pattern()
        True
        >>> TRiP98FilePath("file.txt", pt.DosCube)._is_name_pattern()
        True
        >>> TRiP98FilePath("filephys.txt", pt.DosCube)._is_name_pattern()
        True
        >>> TRiP98FilePath("filephys.dos", pt.DosCube)._is_name_pattern()
        False
        >>> TRiP98FilePath("filephys.hed", pt.DosCube)._is_name_pattern()
        False
        >>> TRiP98FilePath("filephys.hed", pt.LETCube)._is_name_pattern()
        True
        >>> TRiP98FilePath("filemlet.hed", pt.LETCube)._is_name_pattern()
        False
        >>> TRiP98FilePath("file.mlet.hed", pt.LETCube)._is_name_pattern()
        False

        :return:
        """
        return not self._is_stem_pattern()

    @property
    def datafile(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("patient1", pt.DosCube).datafile
        'patient1.dos'
        >>> TRiP98FilePath("patient1.dos", pt.DosCube).datafile
        'patient1.dos'
        >>> TRiP98FilePath("patient1.hed", pt.DosCube).datafile
        'patient1.dos'
        >>> TRiP98FilePath("patient1.phys.hed", pt.DosCube).datafile
        'patient1.phys.dos'

        :return:  path to the (unzipped) data file.
        """
        if self.is_valid_header_path():
            return self._filename_without_extension + self.data_file_extension
        elif self.is_valid_datafile_path():
            return self.name
        else:
            return self.name + self.data_file_extension

    @property
    def header(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("patient1", pt.DosCube).header
        'patient1.hed'
        >>> TRiP98FilePath("patient1.dos", pt.DosCube).header
        'patient1.hed'
        >>> TRiP98FilePath("patient1.hed", pt.DosCube).header
        'patient1.hed'
        >>> TRiP98FilePath("patient1.phys.hed", pt.DosCube).header
        'patient1.phys.hed'

        :return:  path to the (unzipped) data file.
        """
        if self.is_valid_header_path():
            return self.name
        elif self.is_valid_datafile_path():
            return self._filename_without_extension + self.header_file_extension
        else:
            return self.name + self.header_file_extension

    @property
    def basename(self):
        """
        >>> import pytrip as pt
        >>> TRiP98FilePath("patient1", pt.DosCube).basename
        'patient1'
        >>> TRiP98FilePath("patient1.dos", pt.DosCube).basename
        'patient1'
        >>> TRiP98FilePath("patient1.dos", pt.CtxCube).basename
        'patient1.dos'
        >>> TRiP98FilePath("patient1.dos.gz", pt.DosCube).basename
        'patient1'
        >>> TRiP98FilePath("patient1.hed", pt.DosCube).basename
        'patient1'
        >>> TRiP98FilePath("patient1.phys.hed", pt.DosCube).basename
        'patient1.phys'

        :return:  path to the (unzipped) data file.
        """
        return self._filename_without_extension


class TRiP98FileLocator(object):
    def __init__(self, name, cube_type):
        self.trip98path = TRiP98FilePath(name, cube_type)

    @property
    def header(self):
        basename = self.trip98path.basename
        files_tried = []
        for gzip_extension in ("", ".gz", ".GZ"):
            for header_extension in (self.trip98path.header_file_extension.lower(),
                                     self.trip98path.header_file_extension.upper()):
                candidate_path = basename + header_extension + gzip_extension
                if os.path.exists(candidate_path):
                    return candidate_path
                else:
                    files_tried.append(candidate_path)
        logger.warning("Tried opening following files: " + " , ".join(files_tried))
        return None

    @property
    def datafile(self):
        basename = self.trip98path.basename
        files_tried = []
        for gzip_extension in ("", ".gz", ".GZ"):
            for datafile_extension in (self.trip98path.data_file_extension.lower(),
                                       self.trip98path.data_file_extension.upper()):
                candidate_path = basename + datafile_extension + gzip_extension
                if os.path.exists(candidate_path):
                    return candidate_path
                else:
                    files_tried.append(candidate_path)
        logger.warning("Tried opening following files: " + " , ".join(files_tried))
        return None


def evaluator(funct, name='funct'):
    """ Wrapper for evaluating a function.

    :params str funct: string which will be parsed
    :params str name: name which will be assigned to created function.

    :returns: function f build from 'funct' input.
    """
    code = compile(funct, name, 'eval')

    def f(x):
        return eval(code, locals())

    f.__name__ = name
    return f


def volume_histogram(cube, voi=None, bins=256):
    """
    Generic volume histogram calculator, useful for DVH and LVH or similar.

    :params cube: a data cube of any shape, e.g. Dos.cube
    :params voi: optional voi where histogramming will happen.
    :returns [x],[y]: coordinates ready for plotting. Dose (or LET) along x, Normalized volume along y in %.

    If VOI is not given, it will calculate the histogram for the entire dose cube.

    Providing voi will slow down this function a lot, so if in a loop, it is recommended to do masking
    i.e. only provide Dos.cube[mask] instead.
    """

    if voi is None:
        mask = None
    else:
        vcube = voi.get_voi_cube()
        mask = (vcube.cube == 1000)

    _xrange = (0.0, cube.max()*1.1)
    _hist, x = np.histogram(cube[mask], bins=bins, range=_xrange)
    _fhist = _hist[::-1]  # reverse histogram, so first element is for highest dose
    _fhist = np.cumsum(_fhist)
    _hist = _fhist[::-1]  # flip back again to normal representation

    y = 100.0 * _hist / _hist[0]  # volume histograms always plot the right edge of bin, since V(D < x_pos).
    y = np.insert(y, 0, 100.0, axis=0)  # but the leading bin edge is always at V = 100.0%

    return x, y
