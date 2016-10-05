#! /usr/bin/env python
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


class ReadGd(object):
    """This class reads .gd files."""

    def __init__(self, filename, exp=False, agr=False, LET=False):
        """ setup of object """
        self.filename = filename
        self.head = []
        self.legend = []
        self.indata = []
        self.xdata = []
        self.data = []
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""
        self.h = 0
        self.export_Bool = exp
        self.agr = agr
        self.LET_Bool = LET

        print('\n# Running the conversion scrip gd2dat.py')
        for par in sys.argv:
            if par == 'agr':
                self.agr = True
                self.export_Bool = True
            elif par == 'LET':
                self.LET_Bool = True
            elif par[:3] == 'exp':
                self.export_Bool = True

        if self.filename is None:
            print("No file name has been specified")
        else:
            print("# Reading the file " + self.filename + " with gd2dat.py")
            self.read()

    def read(self):
        if os.path.isfile(self.filename) is False:
            raise IOError("Could not find file " + self.filename)

        gd_file = open(self.filename, 'r')
        gd_lines = gd_file.readlines()
        gd_file.close()

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

            if line[s] == 'x' or line[s] == 'X':
                line_len = len(line)
                self.xlabel = line[s + 2:line_len - 1]

            elif line[s] == 'y' or line[s] == 'Y':
                line_len = len(line)
                self.ylabel = line[s + 2:line_len - 1]
            elif line[s] == 'h' \
                    or line[s] == 'H' \
                    or line[s] == 'a' \
                    or line[s] == 'A':
                line_len = len(line)
                p = 0
                while p < line_len:
                    l2 = line[p:p + 2]
                    legend_Bool = False
                    if l2 == 'x ' or l2 == 'X ':
                        self.head.append('x')
                        self.h += 1
                        self.legend.append("x")
                        legend_Bool = False
                    elif l2 == 'y(' or l2 == 'Y(':
                        self.head.append('y')
                        self.h += 1
                        legend_Bool = True
                    elif l2 == 'm(' or l2 == 'M(':
                        self.head.append('m')
                        self.h += 1
                        legend_Bool = True
                    elif l2 == 'n(' or l2 == 'N(':
                        self.head.append('n')
                        self.h += 1
                        legend_Bool = True
                    elif l2 == 'l(' or l2 == 'L(':
                        self.head.append('l')
                        self.h += 1
                        legend_Bool = True

                    if legend_Bool:
                        legend_Bool = False
                        t = p + 1
                        while t < line_len:
                            if line[t] == ')':
                                self.legend.append(line[p + 2:t])
                                if (self.LET_Bool):
                                    if line[p + 2:t] == "mb^DoseLETme^":
                                        self.head[self.h - 1] = "y"
                                        self.legend[self.h - 1] = "DoseLET"
                                    if line[p + 2:t] == "mb^LETme^":
                                        self.head[self.h - 1] = "y"
                                        self.legend[self.h - 1] = "FluenceLET"

                                p = t
                                break
                            else:
                                t += 1
                    p += 1

            elif line[s] == 'n' or line[s] == 'N':
                break
            elif line[s].isdigit() or line[s] == '-':

                line_len = len(line)
                #                print len(line)     debug line
                #                print line[:line_len-1]    debug line
                linelist = (line[:line_len - 1]).split(' ')
                # print linelist     debug line
                self.indata.append(linelist)

        num = 0
        while num < self.h:
            ydat = []
            if self.head[num] == "x":
                for ele in self.indata:
                    self.xdata.append(ele[num])
                    ydat.append(float(ele[num]))
                self.data.append(ydat)
            elif self.head[num] == 'y' or self.head[num] == 'm' or self.head[num] == 'n':  # normal data
                for ele in self.indata:
                    ydat.append(float(ele[num]))
                self.data.append(ydat)
            num += 1

        if self.export_Bool:
            self.export()

    def export(self):

        len_file_name = len(self.filename)
        out_file_name = self.filename[:len_file_name - 2]
        if self.agr:
            out_file_name += "agr"
            print('# Writing data in a ".agr" file fragment: ' + out_file_name)
        else:
            out_file_name += "dat"
            print('# Writing data in a ".dat" file: ' + out_file_name)

        out_file = open(out_file_name, 'w')

        if self.LET_Bool:
            print('# Exporting also LET data  ')

        if self.agr:
            header = "# Grace project file\n# \n@version 50122 \n"
            #            header += "@page size 842, 595 \n@page scroll 5% \n"
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

        #        print self.h
        #        print self.head
        #        print self.legend
        #        print "length self.data ", len(self.data)
        #        print "length self.xdata ", len(self.xdata)
        #        exit()

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

            elif self.head[num] == 'y' or self.head[num] == 'm':  # normal data

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
                #                sys.stdout.write(str_out)
                out_file.write(str_out)

                str_out = '@    s' + str(counter) + ' comment "'
                if not self.agr:
                    str_out = '# ' + str_out
                str_out += self.filename + ' "\n'
                for i, ele in enumerate(self.data[num]):
                    str_out += self.xdata[i] + ' ' + str(ele) + '\n'
                # sys.stdout.write(str_out)
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
                # sys.stdout.write(str_out)
                out_file.write(str_out)

                counter += 1

# end special handling for 'm'

if __name__ == '__main__':
    ReadGd(sys.argv[1], LET=True, exp=True)
