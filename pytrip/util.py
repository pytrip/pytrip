#
#    Copyright (C) 2010-2018 PyTRiP98 Developers.
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

logger = logging.getLogger(__name__)


def get_class_name(item):
    """
    :returns: name of class of 'item' object.
    """
    return item.__class__.__name__


class TRiP98FilePath(object):
    """
    Helper class which deals with filename discovery.
    It helps in getting stem part of the filename (i.e. patient name) from the filename which includes file extension.
    It can also construct full filename when only stem part is provided.
    Cube type can be specified in the class constructor by providing object of cube class (or the class itself).

    TRiP98 files (cubes) are named according to one of the following pattern:

    A) STEM pattern

    STEM + SUFFIX + . + hed  (header file)
    STEM + SUFFIX + . + EXTENSION  (data file)

    STEM is a core part of the path provided by user (i.e. patient name), it can end with DOT (.)
    SUFFIX is used to distinguish between quantities saved in a cube (i.e. PHYS, BIO, RBE, SVV, MLET, DOSEMLET)

    B) NAME pattern

    Here suffix is omitted and we have no information about quantity saved in a cube.
    (i.e. we can save LET cube into the files (1.hed + 1.dos); then upon reading we will not know from the filename
    whether it is a LET or DOS cube).

    NAME + . + hed (header file)
    NAME + . + EXTENSION (data file)

    Each of the files can also be gziped packed and end with .gz

    See http://bio.gsi.de/DOCS/TRiP98/PRO/DOCS/trip98cmddose.html for more details.

    """

    def __init__(self, name, cube_type):
        """
        Creates a helper class to deal with TRiP98 filenames.
        It digest full or partial filename. Header and datafile extensions are extracted from cube_type
        parameter, as each of TRiP98 cubes has a field which holds allowed extensions.
        Each cube can store different parameter (i.e. DosCube can store RBE and dose (PHYS) data.)
        List of allowed parameters is also encoded in the TRiP98 cube classes.
        :param name: full filename to header or datafile (i.e. pat19998.ctx) or part of it (i.e. pat19998).
        :param cube_type: object which identifies TRiP98 cube, i.e. object of CtxCube, DosCube or LETCube.
        """
        self.name = name
        self.cube_type = cube_type
        self.data_file_extension = self.cube_type.data_file_extension
        self.header_file_extension = self.cube_type.header_file_extension
        self.allowed_suffix = self.cube_type.allowed_suffix

    def is_valid_header_path(self):
        """
        Checks if filename corresponds to valid header filename
        (i.e. it it ends with .hed or .HED).
        >>> import pytrip as pt
        >>> TRiP98FilePath("file.txt", pt.CtxCube).is_valid_header_path()
        False
        >>> TRiP98FilePath("file.txt.hed", pt.DosCube).is_valid_header_path()
        True
        >>> TRiP98FilePath("file.HED", pt.CtxCube).is_valid_header_path()
        True
        >>> TRiP98FilePath("file.DOS", pt.DosCube).is_valid_header_path()
        False
        >>> TRiP98FilePath("file.hed", pt.Cube).is_valid_header_path()
        True
        >>> TRiP98FilePath("file.hed", pt.LETCube).is_valid_header_path()
        True
        >>> TRiP98FilePath("file", pt.LETCube).is_valid_header_path()
        False

        :return: True if filename denotes header, False otherwise.
        """

        # if by some wild chance subclass of pt.Cube doesn't specify header file extension, we return False
        if not self.header_file_extension:
            return False

        # check lower and uppercase extensions
        return self._ungzipped_filename.endswith(
            (self.header_file_extension.lower(),
             self.header_file_extension.upper()))

    def is_valid_datafile_path(self):
        """
        Checks if filename corresponds to valid datafile.
        If wanted to get information about DosCube or LETCube, it checks if file extension is .dos,
        for CtxCubes it checks for .ctx extension.

        It doesn't check if filename contains a valid suffix (dosemlet or mlet for LETCube, PHYS, etc... for DosCube.
        Such type of check can be done using is_valid_cube_type() method.

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

        :return: True if filename denotes datafile, False otherwise.
        """

        # check if corresponding cube class has information about data file extension
        if not self.data_file_extension:
            return False

        # check lower and uppercase extensions
        correct_extension = self._ungzipped_filename.endswith((self.data_file_extension.lower(),
                                                               self.data_file_extension.upper()))

        # check if cube type can be identified using suffix part
        compatible_suffix = self.is_valid_cube_type()

        return correct_extension and compatible_suffix

    def is_valid_cube_type(self):
        no_suffix = self.suffix is None
        return no_suffix or (self.suffix in self.allowed_suffix)

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

        # remove .gz if present
        _ungzipped_filename = self._ungzipped_filename

        # check lower and uppercase extensions
        # check if header extension present, if yes, drop it
        if self.header_file_extension and _ungzipped_filename.endswith(
            (self.header_file_extension.lower(),
             self.header_file_extension.upper())):
            return self._ungzipped_filename[:-len(self.header_file_extension)]

        # check if datafile extension present, if yes, drop it
        if self.data_file_extension and _ungzipped_filename.endswith(
            (self.data_file_extension.lower(),
             self.data_file_extension.upper())):
            return _ungzipped_filename[:-len(self.data_file_extension)]

        # no extensions found, return only unzipped filename
        return _ungzipped_filename

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
        if self.is_valid_datafile_path():
            return self.name
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
        if self.is_valid_datafile_path():
            return self._filename_without_extension + self.header_file_extension
        return self.name + self.header_file_extension

    @property
    def basename(self):
        """
        Returns the basename, without leading directory or trailing suffixes.

        >>> import pytrip as pt
        >>> TRiP98FilePath("foobar/patient1", pt.DosCube).basename
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
        # return self._filename_without_extension
        return os.path.basename(self._filename_without_extension)

    @property
    def dir_basename(self):
        """
        Returns the basename including the leading directory.

        >>> import pytrip as pt
        >>> TRiP98FilePath("foobar/patient1", pt.DosCube).dir_basename
        'foobar/patient1'
        >>> TRiP98FilePath("patient1.dos", pt.DosCube).dir_basename
        'patient1'
        >>> TRiP98FilePath("patient1.dos", pt.CtxCube).dir_basename
        'patient1.dos'
        >>> TRiP98FilePath("patient1.dos.gz", pt.DosCube).dir_basename
        'patient1'
        >>> TRiP98FilePath("patient1.hed", pt.DosCube).dir_basename
        'patient1'
        >>> TRiP98FilePath("patient1.phys.hed", pt.DosCube).dir_basename
        'patient1.phys'

        :return:  path + basename to the (unzipped) data file.
        """
        # return self._filename_without_extension
        return self._filename_without_extension


class TRiP98FileLocator(object):
    def __init__(self, name, cube_type):
        self.trip98path = TRiP98FilePath(name, cube_type)

        # construct list of suffixes to check
        # they may or may not include leading dot (i.e. ".dosemlet" or "dosemlet")
        # suffix may also be lower or UPPERCASE
        self.list_of_suffixes_to_check = []
        for optional_dot in ('', '.'):
            for suffix in self.trip98path.allowed_suffix:
                self.list_of_suffixes_to_check.append(optional_dot + suffix.lower())
                self.list_of_suffixes_to_check.append(optional_dot + suffix.upper())
        self.list_of_suffixes_to_check.append("")

    @property
    def header(self):
        """
        Calculates the path to the header file existing on the disk.
        Assuming that in the filesystem files "a.hed" and "c.ctx" are present it will work like following:

        >>> import pytrip as pt
        >>> TRiP98FileLocator("tests/res/TST003/tst003000", pt.CtxCube).header
        'tests/res/TST003/tst003000.hed'

        It can also work well with gzipped filenames:
        >>> ungzipped_file = "tests/res/TST003/tst003012.dosemlet.hed"
        >>> os.path.exists(ungzipped_file)
        False

        >>> gzipped_file = ungzipped_file + ".gz"
        >>> gzipped_file
        'tests/res/TST003/tst003012.dosemlet.hed.gz'
        >>> os.path.exists(gzipped_file)
        True

        FileLocator can automatically find requested file by adding suffix dosemlet which is appropriate to
        requested cube (note that the returned path points to the file which exists on the disk).
        >>> TRiP98FileLocator('tests/res/TST003/tst003012', pt.LETCube).header
        'tests/res/TST003/tst003012.dosemlet.hed.gz'

        After changing the cube to DosCube we get the path to another file (also located on the disk).
        >>> TRiP98FileLocator('tests/res/TST003/tst003012', pt.DosCube).header
        'tests/res/TST003/tst003012.hed'

        :return: path to the header file which exists on the filesystem or None if not found
        """
        dir_basename = self.trip98path.dir_basename
        files_tried = []
        logger.info("Locating : " + self.trip98path.name + " as " + str(self.trip98path.cube_type))

        for suffix in self.list_of_suffixes_to_check:
            for gzip_extension in ("", ".gz", ".GZ"):
                for header_extension in (self.trip98path.header_file_extension.lower(),
                                         self.trip98path.header_file_extension.upper()):
                    candidate_path = dir_basename + suffix + header_extension + gzip_extension
                    if os.path.exists(candidate_path):
                        logger.info("Found " + candidate_path)
                        return candidate_path
                    files_tried.append(candidate_path)
        logger.warning("Checking following files: " + " , ".join(files_tried) + ". None of them exists.")
        return None

    @property
    def datafile(self):
        """
        It works exactly in the same way as header method in TRiP98FileLocator class,
        but returns path to the datafile.

        :return: path to the data file which exists on the filesystem or None if not found
        """
        dir_basename = self.trip98path.dir_basename
        files_tried = []
        logger.info("Locating : " + self.trip98path.name + " as " + str(self.trip98path.cube_type))
        for suffix in self.list_of_suffixes_to_check:
            for gzip_extension in ("", ".gz", ".GZ"):
                for datafile_extension in (self.trip98path.data_file_extension.lower(),
                                           self.trip98path.data_file_extension.upper()):
                    candidate_path = dir_basename + suffix + datafile_extension + gzip_extension
                    if os.path.exists(candidate_path):
                        return candidate_path
                    files_tried.append(candidate_path)
        logger.warning("Checking following files: " + " , ".join(files_tried) + ". None of them exists.")
        return None


def human_readable_size(num, suffix="B"):
    """
    Convert number of bytes to human readable form
    for example 35489 -> 34.66KiB
    :param num: number of bytes to convert
    :param suffix: suffix to use (default B (byte))
    :return: string with number in human readable form
    """
    # https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            number = "{:.2f}".format(num).rstrip('0').rstrip('.')
            return "{:s}{:s}{:s}".format(number, unit, suffix)
        num /= 1024.0
    number = "{:.2f}".format(num).rstrip('0').rstrip('.')
    return "{:s}Yi{:s}".format(number, suffix)


def get_size(start_path="."):
    """
    Calculates size of directory (with subdirectories)
    :param start_path: path to directory
    :return: size of directory in bytes
    """
    # https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size
