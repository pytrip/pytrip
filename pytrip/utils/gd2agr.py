#! /usr/bin/env python
"""
    converts gd files to xmgrace files in .agr format.
    06.10.2011 Armin Lühr
    ---
    comments:
    still limited to .gd files with a single y column
"""

import sys
import os


class ReadGd(object):
    """read file"""

    def __init__(self, filename):

        if os.path.isfile(filename) is False:
            raise IOError("Could not find file " + filename)

        gd_file = open(filename, 'r')
        gd_lines = gd_file.readlines()
        gd_file.close()

        header = "# Grace project file\n# \n@version 50122 \n"
        header += "@page size 842, 595 \n@page scroll 5% \n"
        header += "@page inout 5% \n@link page off \n"
        sys.stdout.write(header)

        line = gd_lines[0]
        line_len = len(line)
        string = '@    title  "' + line[:line_len - 1] + '"\n'
        string += '@    legend char size 1.250000\n'
        sys.stdout.write(string)

        count = -1

        for line in gd_lines[1:]:
            s = 0
            while s < 10:
                if line[s] == ' ':
                    s += 1
                else:
                    break

            if line[s].isdigit() or line[s] == '-':
                string = line

            elif line[s] == 'x' or line[s] == 'X':
                line_len = len(line)
                string = '@    xaxis  label "' + line[s + 2:line_len - 1]
                string += ' "\n@    xaxis  label char size 1.500000\n'
                string += '@    xaxis  ticklabel char size 1.250000\n'

            elif line[s] == 'y' or line[s] == 'Y':
                line_len = len(line)
                string = '@    yaxis  label "' + line[s + 2:line_len - 1]
                string += ' "\n@    yaxis  label char size 1.500000\n'
                string += '@    yaxis  ticklabel char size 1.250000\n'

            elif line[s] == 'h' \
                    or line[s] == 'H' \
                    or line[s] == 'a' \
                    or line[s] == 'A' \
                    or line[s] == 'n' \
                    or line[s] == 'N':
                count += 1
                string = '\n@    s' + str(count) + ' comment "' + filename + ' "\n'
                line_len = len(line)
                p = s + 1

                while p < line_len:
                    if line[p] == 'y' or line[p] == 'Y':
                        t = p + 1
                        while t < line_len:
                            if line[t] == ')':
                                string += '@    s' + str(count) + ' legend  "'
                                string += line[p + 2:t] + '"\n'
                                p = line_len
                                break
                            else:
                                t += 1
                    else:
                        p += 1

                string += '#' + line
            else:
                string = "#" + line

            sys.stdout.write(string)


if __name__ == '__main__':
    ReadGd(sys.argv[1])
