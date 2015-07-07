#! /usr/bin/env python
from rst_read import *

import sys
from pylab import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-s", "--submachine", dest="subm",
                  help="Select submachine to plot.", metavar="int")
parser.add_option("-f", "--factor", dest="fac",
                  help="Factor for scaling the blobs. Default is 1000.", metavar="int")
(options, args) = parser.parse_args()

file = args[0]

sm = 1 # default
fac = 1000 
if options.subm != None:
    sm = int(options.subm)
if options.fac != None:
    fac = int(options.fac)

a = RstfileRead(file)

# convert data in submachine to a nice array
b = a.submachine[sm]
print "Submachine: ", sm, " - Energy:", b.energy, "MeV/u"
cc = array(b.particles)

cc = cc / cc.max() * fac

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(b.xpos, b.ypos, c=cc, s=cc, alpha=0.75)
ylabel("mm")
xlabel("mm")    

grid(True)

plt.show()
