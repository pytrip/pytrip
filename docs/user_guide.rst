.. _user_guide:

===================
PyTRiP User's Guide
===================

.. rubric:: pytrip object model, description of classes, examples



Using PyTRiP as a library
=========================

The full potential of PyTRiP is exposed when using it as a library.

Using the `dir()` and `help()` methods, you may explore what functions are available, check also the index and module tables found in this documentation. 

CT and Dose data are handled by the "CtxCube" and "DosCube" classes, respectively. Structures (volume of interests) are handled by the VdxCube class.
For instance, when a treatment plan was made the resulting 3D dose distribution (and referred to as a "DosCube").

    >>> import pytrip as pt
    >>> dc = pt.DosCube()
    >>> dc.read("foobar.dos")

You can display the entire doscube by simply printing its string
(str or repr) value::

    >>> dc
    ....

We recommend you to take a look at the :doc:`Examples </examples>` and browse the :ref:`modindex` page.

Converters
==========

A few converters based on PyTRiP are supplied as well. These converters are:

:trip2dicom.py:
   converts a Voxelplan formatted file to a Dicom file.
   
:dicom2trip.py:
   converts a Dicom file to a Voxelplan formatted file.
   
:cubeslice.py:
   Generates .png files for each slice found in the given cube.
   
:gd2dat.py:
   Converts a GD formatted plot into a stripped ASCII-file

:gd2agr.py:
   Converts a GD formatted plot into a a `xmgrace <http://plasma-gate.weizmann.ac.il/Grace/>`_ formatted plot.

:rst2sobp.py:
   Converts a raster scan file to a file which can be read by FLUKA or SHIELD-HIT12A.
