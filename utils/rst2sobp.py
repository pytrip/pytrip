#! /usr/bin/env python
from rst_read import *

import sys

file = sys.argv[1]

#a=RstfileRead("cub000101.rst")
a = RstfileRead(file)
fout = open("sobp.dat",'w')
for i in range(a.submachines):
    b = a.submachine[i]
    for j in range(len(b.xpos)):
        fout.writelines("%-10.6f%-10.2f%-10.2f%-10.2f%-10.4e\n" % (b.energy/1000.0, b.xpos[j]/10.0, b.ypos[j]/10.0, b.focus/10.0, b.particles[j]))
fout.close()

