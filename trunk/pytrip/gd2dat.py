#! /usr/bin/env python
""" convert gd files to xmgrace readable ascii data.
"""


import sys
import os
import glob

def is_float(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

class ReadGd(object):
    'read file'

    def __init__(self, fn):
        fn_list = glob.glob(fn)
        for filename in fn_list:
            if os.path.isfile(filename) is False:
                raise IOError,  "Could not find file " + filename
            print filename
            gd_file = open(filename, 'r')
            base,ext = os.path.splitext(filename)
            out_file =open(base+'.dat','w')

            gd_lines = gd_file.readlines()
            gd_file.close()
            first = True
            ignore_rest = False
            for line in gd_lines:
                #if not(line[0].isdigit()):
                fff = line.split()
                if len(fff) > 0:
                    if not(is_float(fff[0])):
                        string = "#" + line
                        if first == False:
                            ignore_rest = True
                    else:
                        first = False
                        if ignore_rest:
                            string = "#" + line
                        else:
                            string = line

                #sys.stdout.write(string)
                out_file.write(string)
            out_file.close()
            gd_file.close()




if __name__ == '__main__':

    ReadGd(sys.argv[1])
