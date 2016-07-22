#! /usr/bin/env python

import sys

from pytrip.utils.rst_read import RstfileRead


def main(args=sys.argv[1:]):
    file = args[0]
    a = RstfileRead(file)
    fout = open("sobp.dat", 'w')
    for i in range(a.submachines):
        b = a.submachine[i]
        for j in range(len(b.xpos)):
            fout.writelines("%-10.6f%-10.2f%-10.2f%-10.2f%-10.4e\n" % (b.energy / 1000.0, b.xpos[j] / 10.0, b.ypos[j] /
                                                                       10.0, b.focus / 10.0, b.particles[j]))
    fout.close()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
