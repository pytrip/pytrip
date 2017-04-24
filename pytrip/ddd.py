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
This module provides the DDD class for handling depth-dose curve kernels for TRiP98.
"""
import logging
import os
import time
from glob import glob
from numbers import Number

try:
    from UserDict import UserDict  # python 2
except ImportError:
    from collections import UserDict  # python 3

import numpy as np

from pytrip.res.interpolate import RegularInterpolator

logger = logging.getLogger(__name__)


class DDD(UserDict):  # UserDict allows us to use DDD objects as pure dictionaries with our own new methods
    """
    Represents set of DDD files.
    Internally it is a dictionary with keys being energies and values being DDDFile objects.
    """

    def dose(self, energy, r, z):
        pass

    def read(self, path):
        """
        Read DDD data from path (single file, directory or glob expression)
        :param path:
        :return:
        """
        if os.path.isfile(path):  # read a single file
            ddd = DDDFileReader(path)
            self[ddd.energy] = ddd
        elif os.path.isdir(path):  # single directory
            glob_expression = os.path.join(path, "*.ddd")
            self._read_from_glob_expression(glob_expression)
        else:  # treat as glob expression (i.e. "/home/test/trip/a_*.ddd") by default
            self._read_from_glob_expression(path)

    def write(self, directory, prefix="file_"):
        """
        Save DDD data to a directory
        :param directory: place to save DDD filess
        :param prefix: optional prefix, by default "file_"
        :return:
        """
        suffix = ".ddd"
        for energy, ddd in self.items():
            path = os.path.join(directory, "{:s}{:g}{:s}".format(prefix, energy, suffix))
            ddd.write(path)

    def energies(self):
        """
        List of initial energies related to DDD files loaded
        :return:
        """
        return self.keys()

    def _read_from_glob_expression(self, glob_expr):
        for item in glob(glob_expr):
            ddd = DDDFileReader.read(item)
            if ddd.energy in self.keys():
                raise Exception("DDD file with energy {:g} already present in collection".format(ddd.energy))
            self.data[ddd.energy] = ddd


class DDDFile:
    """
    Represents a single DDD file
    """
    def __init__(self, projectile=None, energy=None, data=None):
        # default members
        self.filetype = 'ddd'
        self.fileversion = '19980520'
        self.filedate = time.strftime('%c'),  # Locale's appropriate date and time representation
        self.material = 'H2O'
        self.composition = 'H2O'
        self.density = 1.0  # floating point number, density of material

        # values to be filled by user
        self.projectile = projectile
        self.energy = energy
        self.data = data

        # B-spline representation of 1-D curve
        self._dedz_interp_fun = self.__interp_fun(self.de_dz)
        self._fwhm1_interp_fun = self.__interp_fun(self.fwhm1)  # may be None if FWHM1 data missing
        self._factor_interp_fun = self.__interp_fun(self.factor)  # may be None if factor data missing
        self._fwhm2_interp_fun = self.__interp_fun(self.fwhm2)  # may be None if FWHM2 data missing

    def dose(self, r_cm, z_cm):
        if self.data.shape[1] < 3:  # no information about FWHM in the files
            return self.de_dz(z_cm)
        elif self.data.shape[1] < 5:  # single gaussian data
            return GaussianModel.single_gauss_MeV_g(
                r_cm=r_cm,
                amp_MeV_cm_g=self.de_dz(z_cm),
                sigma_cm=self.fwhm1(z_cm) / GaussianModel.fwhm_to_sigma
            )
        else:  # double gaussian data
            return GaussianModel.double_gauss_MeV_g(
                r_cm=r_cm,
                amp_MeV_cm_g=self.de_dz(z_cm),
                sigma1_cm=self.fwhm1(z_cm) / GaussianModel.fwhm_to_sigma,
                weight=self.factor(z_cm),
                sigma2_cm=self.fwhm2(z_cm) / GaussianModel.fwhm_to_sigma,
            )

    def read(self, filename):
        ddd = DDDFileReader.read(filename)
        # update self with data read from the file
        for item in vars(ddd):
            setattr(self, item, getattr(ddd, item))

    def write(self, filename):
        DDDFileWriter.write(self, filename)

    # data section - Z axis getter
    @property
    def z(self):
        if self.data is not None:
            return self.data[:, 0]
        return None

    def de_dz(self, z_cm=None):
        """
        Evaluate dE/dz (central axis dose) from DDD set
        If z_cm is provided then perform B-spline interpolation to get intermediate values
        If z_cm is None, return full table of dE/dz from DDD file
        :param z_cm:
        :return:
        """
        if self.data is None:
            return None
        if z_cm is None:
            return self.data[:, 1]
        else:
            result = self._dedz_interp_fun(x=z_cm)
            return result

    def fwhm1(self, z_cm):
        """
        Evaluate FWHM1 from DDD set
        If z_cm is provided then perform B-spline interpolation to get intermediate values
        If z_cm is None, return full table of FWHM1 from DDD file
        :param z_cm:
        :return:
        """
        if self.data.shape[1] < 3:
            return None
        if z_cm is None:
            return self.data[:, 2]
        else:
            result = self._fwhm1_interp_fun(x=z_cm)
            return result

    def factor(self, z_cm):
        """
        Evaluate factor from DDD set
        If z_cm is provided then perform B-spline interpolation to get intermediate values
        If z_cm is None, return full table of factor from DDD file
        :param z_cm:
        :return:
        """
        if self.data.shape[1] < 4:
            return None
        if z_cm is None:
            return self.data[:, 3]
        else:
            result = self._factor_interp_fun(x=z_cm)
            return result

    def fwhm2(self, z_cm):
        """
        Evaluate FWHM2 from DDD set
        If z_cm is provided then perform B-spline interpolation to get intermediate values
        If z_cm is None, return full table of FWHM2 from DDD file
        :param z_cm:
        :return:
        """
        if self.data.shape[1] < 5:
            return None
        if z_cm is None:
            return self.data[:, 4]
        else:
            result = self._fwhm2_interp_fun(x=z_cm)
            return result

    # validate energy
    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, e):
        if isinstance(e, Number):  # check if a number
            if e < 0.0:
                raise Exception("Energy should be a positive number")
        self._energy = e

    # validate density
    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, d):
        if isinstance(d, Number):  # check if a number
            if d < 0.0:
                raise Exception("Density should be a positive number")
        self._density = d

    def update_bspline(self):
        """
        Calculate B-spline representation of dE/dz, fwhm1, factor and FWHM2
        :return:
        """
        self._dedz_interp_fun = self.__interp_fun(self.de_dz)
        self._fwhm1_interp_fun = self.__interp_fun(self.fwhm1)
        self._factor_interp_fun = self.__interp_fun(self.factor)
        self._fwhm2_interp_fun = self.__interp_fun(self.fwhm2)

    def __interp_fun(self, y_data):
        """
        Calculate 1-D B-spline on a dataset with X coordinates being self.z and Y coordinates provided as param
        :param y_data:
        :return:
        """
        result = None
        if y_data is not None:
            result = RegularInterpolator(
                x=self.z,
                y=y_data,
                kind='spline'
            )
        return result


class GaussianModel:
    """
    TODO
    """

    def __init__(self):
        pass

    fwhm_to_sigma = np.sqrt(8.0 * np.log(2))

    @classmethod
    def single_gauss_MeV_g(cls, r_cm, amp_MeV_cm_g, sigma_cm):
        """
        TODO
        :param r_cm:
        :param amp_MeV_cm_g:
        :param sigma_cm:
        :return:
        """
        return amp_MeV_cm_g / (2.0 * np.pi * sigma_cm) * np.exp(-r_cm ** 2 / (2.0 * sigma_cm ** 2))

    @classmethod
    def double_gauss_MeV_g(cls, r_cm, amp_MeV_cm_g, sigma1_cm, weight, sigma2_cm):
        """
        TODO
        :param r_cm:
        :param amp_MeV_cm_g:
        :param sigma1_cm:
        :param weight:
        :param sigma2_cm:
        :return:
        """
        return weight * cls.single_gauss_MeV_g(r_cm, amp_MeV_cm_g, sigma1_cm) + \
            (1.0 - weight) * cls.single_gauss_MeV_g(r_cm, amp_MeV_cm_g, sigma2_cm)


class DDDFileReader:
    # list of allowed tags in DDD file, extracted from members of DDD class
    def __init__(self):
        pass

    allowed_tags = ('filetype', 'fileversion', 'filedate', 'material', 'composition', 'density', 'projectile', 'energy')

    @staticmethod
    def _is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    @classmethod
    def _read_metadata(cls, filename):
        """
        get metadata from file and save as dictionary
        keys are tag names
        values are extracted from the part of line which is following tag name
        for special tag named `ddd` line number is saved as a value
        :param filename: path to the filename
        :return: dictionary with metadata (empty if file is missing)
        """
        metadata_dict = {}

        with open(filename, 'r') as f:
            for line_no, line in enumerate(f.readlines()):

                # check line starting with non-allowed characters
                if not line.startswith(('!', '#', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-')):
                    raise Exception("Line starting with incompatible character in {:s}, "
                                    "line no: {:d}".format(filename, line_no))

                # split the line using whitespace as separator
                line_items = line.split()

                # check for empty lines
                if not line_items:
                    raise Exception("Empty line found in {:s}, line no: {:d}".format(filename, line_no))

                # line not empty, get candidate for tag
                first_item = line_items[0]

                if first_item.startswith('!'):  # line with tag

                    # get tag
                    tag = first_item[1:]

                    # check if tag is not empty
                    if not tag:
                        raise Exception("Empty tag found in {:s}, line no: {:d}".format(filename, line_no))

                    # check if tag is allowed
                    if tag not in cls.allowed_tags and tag != 'ddd':
                        raise Exception("Incompatible tag {:s} found in {:s}, line no: {:d}".format(
                            tag,
                            filename,
                            line_no))

                    # check if tag is doubled
                    if tag in metadata_dict:
                        raise Exception("Doubled {:s} tag found in {:s}, line no: {:d}".format(
                            tag,
                            filename,
                            line_no))

                    # get value (removing white spaces from left and right part)
                    value = line[len(first_item):].rstrip().lstrip()

                    # put tag and value in the dictionary
                    if tag in cls.allowed_tags:
                        metadata_dict[tag] = value
                    else:
                        metadata_dict[tag] = line_no

                elif line.startswith('#'):  # comment line
                    pass
                elif cls._is_number(first_item):  # looks like line with data
                    pass
                else:
                    raise Exception("Unknown line found in {:s}, line no: {:d}".format(filename, line_no))
        return metadata_dict

    @classmethod
    def read(cls, filename):
        """Construct an DDD object from data in a file."""

        # read metadata from file and save in metadata_dict dictionary
        # keys are tag names
        # values are extracted from the part of line which is following tag name
        # for special tag named `ddd` line number is saved as a value
        metadata_dict = cls._read_metadata(filename)

        try:
            data_part_line_no = metadata_dict['ddd']
        except KeyError:
            raise Exception("No data section in file {:s}".format(filename))

        # read data from data part of the file
        data = np.loadtxt(filename,
                          comments=('#', '!'),  # skip tag and comment lines
                          skiprows=data_part_line_no,  # skip block with tags
                          ndmin=2,  # ensure loading at least 2D array (columns and rows)
                          )

        if data.shape[1] < 2 or data.shape[1] > 5:
            raise Exception("File {:s} should have between 2 and 5 columns of data")

        # conversion from numbers to floats for specific items
        metadata_dict['energy'] = float(metadata_dict['energy'])
        metadata_dict['density'] = float(metadata_dict['density'])

        ddd = DDDFile(data=data)
        for tag, value in metadata_dict.items():
            setattr(ddd, tag, value)

        return ddd


class DDDFileWriter:
    def __init__(self):
        pass

    _ddd_header_template = """!filetype    ddd
!fileversion   {fileversion:s}
!filedate      {filedate:s}
!projectile    {projectile:s}
!material      {material:s}
!composition   {composition:s}
!density {density:f}
!energy {energy:f}
#   z[g/cm**2] dE/dz[MeV/(g/cm**2)] FWHM1[g/cm**2] factor FWHM2[g/cm**2]
!ddd"""

    @classmethod
    def write(cls, ddd, filename):
        # prepare header of DDD file
        header = cls._ddd_header_template.format(
            fileversion=ddd.fileversion,
            filedate=ddd.filedate,
            projectile=ddd.projectile,
            material=ddd.material,
            composition=ddd.composition,
            density=ddd.density,
            energy=ddd.energy)

        # write the contents of the files
        np.savetxt(fname=filename,
                   X=ddd.data,
                   delimiter=' ',
                   fmt='%g',
                   comments='',
                   header=header)
