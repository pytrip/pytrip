.. _user_guide:

=================
pytrip User Guide
=================

.. rubric:: pytrip object model, description of classes, examples


Converters
==========

trip2dicom, dicom2trip, cubeslice




pytrip as a library
===================

TODO

A doscube could be created directly...

    >>> import pytrip as pt
    >>> dc = pt.DosCube()
    >>> dc.read("...")

You can display the entire doscube by simply printing its string
(str or repr) value::

    >>> dc
    ....

