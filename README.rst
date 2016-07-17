WHAT IS THIS ?
==============

PyTRiP is a python package for working with TRiP and VIRTUOS/VOXELPLAN files.
It is mainly supposed for batch processing, but an experimental GUI is also included
(see https://github.com/pytrip/pytripgui repo).


HOW TO WORK WITH IT ?
=====================

First, install pytrip98 package. The easiest way is to use pip package manager::

    pip install pytrip98

Following Python code demonstrates PyTRiP capabilities::

    from pytrip import *

    # read a dose cube, divide by 2.0, and write to a new cube:
    d0 = DosCube()
    d0.read("box050001.dos")
    d0 = d0/2.0
    d0.write("out0.dos")

    # sum two dose cubes, write result:
    print "Two half boxes: out.dos"
    d1 = DosCube()
    d2 = DosCube()
    d1.read("box052000.dos")
    d2.read("box053000.dos")
    d = (d1 + d2)
    d.write("out.dos")


    # print minium and maximum value found in cubes
    print d1.cube.min(),d1.cube.max()
    print d0.cube.min(),d0.cube.max()

    # calculate new dose average LET cube
    l1 = LETCube()
    l2 = LETCube()
    l1.read("box052000.dosemlet.dos")
    l2.read("box053000.dosemlet.dos")

    l = ((d1 * l1) + (d2 * l2)) / (d1 + d2)
    l.write("out.dosemlet.dos")


MORE DOCUMENTATION
==================

Full documentation can be found here:
https://pytrip.readthedocs.io/

If you would like to download the code and modify it, read first `this <docs/technical.rst>`__.