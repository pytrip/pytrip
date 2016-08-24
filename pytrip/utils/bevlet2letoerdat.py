#! /usr/bin/env python
""" convert gd files to xmgrace readable ascii data.
"""

import sys
import os


class ReadGd(object):
    '''read file'''

    def __init__(self, filename):

        if os.path.isfile(filename) is False:
            raise IOError("Could not find file " + filename)

        gd_file = open(filename, 'r')
        gd_lines = gd_file.readlines()
        gd_file.close()
        first = True
        ignore_rest = False
        for line in gd_lines:
            if not (line[0].isdigit()):
                string = "#" + line
                if not first:
                    ignore_rest = True
            else:
                first = False
                if ignore_rest:
                    string = "#" + line
                else:
                    string = line

            sys.stdout.write(string)


if __name__ == '__main__':
    ReadGd(sys.argv[1])
