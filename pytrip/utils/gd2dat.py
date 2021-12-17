#! /usr/bin/env python
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
""" Reads .gd files and can convert them into (xmgrace readable) ascii data.

Can be used in the command line

Mandatory arguments:
Name      Type       Default  Description
filename  STRING     ""       File name of .gd file, should be first.

Optional arguments:
Name      Type       Default  Description
exp       STRING     "exp"    if "exp": Export data of .gd into a .dat file
agr       STRING     ""       if "agr": Export data of .gd into a .agr file
LET       STRING     "LET"    if "LET": Export also data entitled "DoseLET"

Example:
1) reading the file foo.gd and exporting it as ascii file foo.dat
python gd2dat.py foo.gd

2) reading the file foo.gd and exporting it as xmgrace file foo.agr
python gd2dat.py foo.gd agr
"""
import sys
import os
import logging
import argparse

import pytrip as pt


class ReadGd(object):
    """This class reads .gd files."""
    # TODO: this class could go into main pytrip/ as well.

    def __init__(self, gd_filename, exp=False, agr=False, let=False):
        """ setup of object """
        self.filename = gd_filename
        self.head = []
        self.legend = []
        self.indata = []
        self.xdata = []
        self.data = []
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""
        self.h = 0
        self.export_bool = exp
        self.agr = agr
        self.let_bool = let

        print('\n# Running the conversion scrip gd2dat.py')
        if self.filename is None:
            print("No file name has been specified")
        else:
            print("# Reading the file " + self.filename + " with gd2dat.py")
            self.read()

    def read(self):
        """ Reads the filename set when constructing the ReadGd class
        """
        if os.path.isfile(self.filename) is False:
            raise IOError("Could not find file " + self.filename)

        with open(self.filename, "r") as gd_file:
            gd_lines = gd_file.readlines()

        line = gd_lines[0]
        line_len = len(line)
        self.title = line[:line_len - 1]

        for line in gd_lines[1:]:
            s = 0
            while s < 10:
                if line[s] == ' ':
                    s += 1
                else:
                    break

            if line[s] in ('x', 'X'):
                line_len = len(line)
                self.xlabel = line[s + 2:line_len - 1]
            elif line[s] in ('y', 'Y'):
                line_len = len(line)
                self.ylabel = line[s + 2:line_len - 1]
            elif line[s] in ('h', 'H', 'a', 'A'):
                line_len = len(line)
                p = 0
                while p < line_len:
                    l2 = line[p:p + 2]
                    legend_bool = False
                    if l2 in ('x ', 'X '):
                        self.head.append('x')
                        self.h += 1
                        self.legend.append("x")
                        legend_bool = False
                    elif l2 in ('y(', 'Y('):
                        self.head.append('y')
                        self.h += 1
                        legend_bool = True
                    elif l2 in ('m(', 'M('):
                        self.head.append('m')
                        self.h += 1
                        legend_bool = True
                    elif l2 in ('n(', 'N('):
                        self.head.append('n')
                        self.h += 1
                        legend_bool = True
                    elif l2 in ('l(', 'L('):
                        self.head.append('l')
                        self.h += 1
                        legend_bool = True

                    if legend_bool:
                        legend_bool = False
                        t = p + 1
                        while t < line_len:
                            if line[t] == ')':
                                self.legend.append(line[p + 2:t])
                                if self.let_bool:
                                    if line[p + 2:t] == "mb^DoseLETme^":
                                        self.head[self.h - 1] = "y"
                                        self.legend[self.h - 1] = "DoseLET"
                                    if line[p + 2:t] == "mb^LETme^":
                                        self.head[self.h - 1] = "y"
                                        self.legend[self.h - 1] = "FluenceLET"

                                p = t
                                break
                            t += 1
                    p += 1

            elif line[s] in ('n', 'N'):
                break
            elif line[s].isdigit() or line[s] == '-':
                line_len = len(line)
                linelist = (line[:line_len - 1]).split(' ')
                self.indata.append(linelist)

        num = 0
        while num < self.h:
            ydat = []
            if self.head[num] == "x":
                for ele in self.indata:
                    self.xdata.append(ele[num])
                    ydat.append(float(ele[num]))
                self.data.append(ydat)
            elif self.head[num] in ('y', 'm', 'n'):  # normal data
                for ele in self.indata:
                    ydat.append(float(ele[num]))
                self.data.append(ydat)
            num += 1

    def export(self, out_file_name=None):
        """ Export data to out_file_name
        :params str out_file_name: full path to output file including file extension
        """
        if out_file_name is None:
            len_file_name = len(self.filename)
            out_file_name = self.filename[:len_file_name - 2]
            if self.agr:
                out_file_name += "agr"
                print('# Writing data in a ".agr" file fragment: ' + out_file_name)
            else:
                out_file_name += "dat"
                print('# Writing data in a ".dat" file: ' + out_file_name)

        with open(out_file_name, "w") as out_file:

            if self.let_bool:
                print('# Exporting also LET data  ')

            if self.agr:
                header = "# Grace project file\n# \n@version 50122 \n"
                # header += "@page size 842, 595 \n@page scroll 5% \n"
                header += "@page inout 5% \n@link page off \n"

                str_out = header + '@    title  "' + self.title + ' "\n'

                str_out += '@    xaxis  label "' + self.xlabel
                str_out += ' "\n@    xaxis  label char size 1.500000\n'
                str_out += '@    xaxis  ticklabel char size 1.250000\n'

                str_out += '@    yaxis  label "' + self.ylabel
                str_out += ' "\n@    yaxis  label char size 1.500000\n'
                str_out += '@    yaxis  ticklabel char size 1.250000\n'

                out_file.write(str_out)

            num = -1
            counter = 0

            while num < self.h - 1:

                num += 1
                # The following feature is still under development
                #            title = self.legend[num]
                #            if (title == "Survival" or title == "SpecDose" \
                #                    or title == "PhysDose" ):
                #                print '# Skip data with the title "' + title + '"'
                #                continue
                #            print self.head[num]  # debug line

                if self.head[num] == "x":
                    continue

                if self.head[num] in ('y', 'm'):  # normal data

                    if self.head[num] == 'm':
                        str_hd = self.legend[num]
                        j = 0
                        for sign in str_hd:
                            if sign == '*':
                                break
                            j += 1
                        _legend = str_hd[j + 1:]
                    else:
                        _legend = self.legend[num]

                    str_out = ' \n'
                    if not self.agr:
                        str_out += '# '
                    str_out += '@    s' + str(counter) + ' legend  "'
                    str_out += _legend + '"\n'
                    out_file.write(str_out)

                    str_out = '@    s' + str(counter) + ' comment "'
                    if not self.agr:
                        str_out = '# ' + str_out
                    str_out += self.filename + ' "\n'
                    for i, ele in enumerate(self.data[num]):
                        str_out += self.xdata[i] + ' ' + str(ele) + '\n'
                    out_file.write(str_out)
                    counter += 1

                else:
                    continue

                # special handling for 'm'
                #
                # A column with the header 'm' should be multiplied with the
                # column to the left of it (cf GD manual).
                #
                if self.head[num] == 'm':

                    str_out = ' \n'
                    if not self.agr:
                        str_out += '# '
                    _legend = self.legend[num]
                    str_out += '@    s' + str(counter) + ' legend  "'
                    str_out += _legend + '"\n'
                    if not self.agr:
                        str_out += '# '
                    str_out += '@    s' + str(counter)
                    str_out += ' comment "' + self.filename + ' "\n'
                    for i, ele in enumerate(self.indata):
                        str_out += self.xdata[i] + ' ' + str((float(ele[num]) * float(ele[num - 1]))) + '\n'
                    out_file.write(str_out)

                    counter += 1

# end special handling for 'm'


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("gd_file", help="location of gd file", type=str)
    parser.add_argument("dat_file", help="location of .dat to write", type=str, nargs='?')
    parser.add_argument("-v", "--verbosity", action='count', help="increase output verbosity", default=0)
    parser.add_argument('-V', '--version', action='version', version=pt.__version__)
    args = parser.parse_args(args)

    gd_data = ReadGd(args.gd_file, let=False, exp=True)
    gd_data.export(args.dat_file)

    return 0


if __name__ == '__main__':
    logging.basicConfig()
    sys.exit(main(sys.argv[1:]))

# TODO add also exporting LET data
