.. _getting_started:

===========================
Getting Started with pytrip
===========================

.. rubric:: Brief overview of pytrip and how to install.

Introduction
==============

PyTRiP is a python package for working with TRiP and VIRTUOS/VOXELPLAN files.

pydicom makes it easy to ...

Here is a simple example of using pytrip98 in an interactive session, to ...

>>> import pytrip as pt
>>> 2+2  # out input
4


pytrip is not a ....


License
=======

pydicom has an GPLv3 `license
<https://github.com/pytrip/pytrip/blob/master/source/GPL_LICENSE.txt>`_.


Installing
==========

As a pure python package, pytrip is easy to install and has no
requirements other than python itself (the NumPy library is recommended,
but is only required if manipulating pixel data).

.. note::
    In addition to the instructions below, pydicom can also be installed
    through the `Python(x,y) <http://www.pythonxy.com/>`_ distribution, which can
    install python and a number of packages [#]_ (including pydicom) at once.


Prerequisites
-------------

  * python 2.7, 3.2 or later
  * `NumPy <http://numpy.scipy.org/>`_ -- TODO
  * `matplotlib <http://google.com/>`_ -- TODO
  * `paramiko <http://google.com/>`_ -- TODO


Python installers can be found at the python web site
(http://python.org/download/).

Installing using pip (all platforms)
----------------------------------------------------
The easiest way to install pydicom is using `pip <https://pypi.python.org/pypi/pip>`_::

    pip install pytrip98

Depending on your python version, there may be some warning messages,
but the install should still be ok.

.. note::
    Pip comes pre-installed with Python newer than 3.4 and 2.?? (for 2.x family)


Using pydicom
=============

Once installed, the package can be imported at a python command line or used
in your own python program with ``import pytrip``.
See the `examples directory
<https://github.com/pytrip/pytrip/tree/examples>`_
for both kinds of uses. Also see the :doc:`User Guide </user_guide>`
for more details of how to use the package.


Support
=======

Bugs can be submitted through the `issue tracker <https://github.com/pytrip/pytrip/issues>`_.
Besides the example directory, cookbook recipes are encouraged to be posted on the
`wiki page <https://github.com/pytrip/pytrip/wiki>`_


Next Steps
==========

To start learning how to use pytrip, see the :doc:`user_guide`.