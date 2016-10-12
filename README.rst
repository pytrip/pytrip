WHAT IS THIS ?
==============

PyTRiP is a python package for working with TRiP and VIRTUOS/VOXELPLAN files.
It is mainly supposed for batch processing, but an experimental GUI is also included
(see https://github.com/pytrip/pytripgui).

**pytrip** provides several command line applications including ``trip2dicom``, ``dicom2trip`` and ``cubeslice``.
They works under Linux, Windows and Mac OSX operating systems
(interpreter of Python programming language has to be also installed).
No programming knowledge is required from user, but basic skills in working with terminal console are needed to use them.


Quick installation guide
------------------------

We recommend that you run a modern Linux distribution, like: **Ubuntu 16.04** or newer, **Debian 9 Stretch** (currently known as testing)
or any updated rolling release (archLinux, openSUSE tumbleweed). In that case, be sure you have **python**
and **python-pip** installed. To get them on Debian or Ubuntu, type being logged in as normal user::

    $ sudo apt-get install python-pip

To automatically download and install the pytrip library, type::

    $ sudo pip install pytrip98

NOTE: the package is named **pytrip98**, while the name of library is **pytrip**.

This command will automatically download and install **pytrip** for all users in your system.

For more detailed instruction, see `installation guide <INSTALL.rst>`__.

To learn how to install pytrip GUI, proceed to following document page: https://github.com/pytrip/pytripgui

Short documentation
-------------------

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

    # load a vdx
    v = VdxCube("", d0)
    v.read("contours.vdx")



More documentation
------------------

Full documentation can be found here:
https://pytrip.readthedocs.io/

If you would like to download the code and modify it, read first `contribution guide <CONTRIBUTING.rst>`__.
