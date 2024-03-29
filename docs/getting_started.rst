.. _getting_started:

===========================
Getting Started with PyTRiP
===========================

.. rubric:: Brief overview of PyTRiP98 and how to install it.

Introduction
==============

PyTRiP98 is a python package for working with files generated by the treatment planning systems
`TRiP98 <http://bio.gsi.de/DOCS/TRiP98/PRO/DOCS/trip98.html>`_ and
`VIRTUOS/VOXELPLAN <https://www.dkfz.de/en/medphys/Therapy_planning_development/Projects/Virtuos.html>`_. Dicom files are also to some extent supported.

PyTRiP will simplify importing and exporting files, processing the data, and also execute TRiP98 locally or remotely.
Thereby it is possible to work with TRiP98 in e.g. a Windows environment, while accessing TRiP98 on a remote server.
PyTRiP enables scripting for large parameters studies of treatment plans, and also more advanced and automized
manipulation than what commercial treatment planning systems might allow.

Let us for instance assume, that one wants (for whatever reason) to reduce all Hounsfield units in a CT cube with a factor of two and write the result into a new file, this can be realized with a few lines of code.

>>> import pytrip as pt
>>> c = pt.CtxCube()  
>>> c.read("tst000001.ctx")  # read a .ctx file

Where the first line imports the pytrip modules, the second line initialized the CtxCube object. The new
object holds (among others) the read() method, which is then being used to load the CT data.
Now let's work with the CT data:

>>> c *= 0.5  # reduce all HUs inside c with a factor of two

and write it to disk.

>>> c.write("out0000001.ctx") # write the new file.

And that all.

We may want to inspect the data, what is the largest and the smalles HU value found in the cube?

>>> print(c.cube.min())
>>> print(c.cube.max())

To see all available methods and attributes, one can run the

>>> dir(c)

command, or read the detailed documentation.

Quick Installation Guide
========================

PyTRiP is available for python 3.6 or later, and can be installed via pip. 

We recommend that you run a modern Linux distribution, like: **Ubuntu 22.04** or newer, **Debian 11**
 or any updated rolling release (archLinux, openSUSE tumbleweed). In that case,
be sure you have **python** and **python-pip** installed.
To get them on Debian or Ubuntu, type being logged in as normal user::

    $ sudo apt-get install python-pip

To automatically download and install the pytrip library, type::

    $ sudo pip install pytrip98

NOTE: the package is named **pytrip98**, while the name of library is **pytrip**.

This command will automatically download and install **pytrip** for all users in your system.

For more detailed instruction, see the :doc:`Detailed Installation Guide </install>`

To learn how to install **pytripgui** graphical user interface, proceed to following document page:
https://github.com/pytrip/pytripgui


Using PyTRiP
============

Once installed, the package can be imported at a python command line or used
in your own python program with ``import pytrip as pt``.
See the `examples directory
<https://github.com/pytrip/pytrip/tree/examples>`_
for both kinds of uses. Also see the :doc:`User's Guide </user_guide>`
for more details of how to use the package.


Support
=======

Bugs can be submitted through the `issue tracker <https://github.com/pytrip/pytrip/issues>`_.
Besides the example directory, cookbook recipes are encouraged to be posted on the
`wiki page <https://github.com/pytrip/pytrip/wiki>`_


Next Steps
==========

To start learning how to use PyTRiP, see the :doc:`user_guide`.


License
=======

PyTRiP98 is licensed under `GPLv3
<https://github.com/pytrip/pytrip/blob/master/source/GPL_LICENSE>`_.
