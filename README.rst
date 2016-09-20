WHAT IS THIS ?
==============

PyTRiP is a python package for working with TRiP and VIRTUOS/VOXELPLAN files.
It is mainly supposed for batch processing, but an experimental GUI is also included
(see https://github.com/pytrip/pytripgui repo).

**mcpartools** provides a command line application called ``generatemc`` which works under Linux operating system
(interpreter of Python programming language has to be also installed).
No programming knowledge is required from user, but basic skills in working with terminal console are needed.


Quick installation guide
------------------------

First be sure to have Python framework installed, then type::

    pip install mcpartools

This command will automatically download and install **mcpartools** for all users in your system.
In case you don't have administrator rights, add ``--user`` flag to ``pip`` command.
In this situation converter will be probably installed in ``~/.local/bin`` directory.

For more detailed instruction, see `installation guide <INSTALL.rst>`__.

Short documentation
-------------------

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


More documentation
------------------

Full documentation can be found here:
https://pytrip.readthedocs.io/

If you would like to download the code and modify it, read first `contribution guide <CONTRIBUTING.rst>`__.