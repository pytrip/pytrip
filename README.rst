===============================
pytrip
===============================

.. image:: https://img.shields.io/pypi/v/pytrip.svg
        :target: https://pypi.python.org/pypi/pytrip
.. image:: https://img.shields.io/travis/pytrip/pytrip.svg
        :target: https://travis-ci.org/pytrip/pytrip


.. image:: https://readthedocs.org/projects/pytrip/badge/?version=latest
        :target: https://readthedocs.org/projects/pytrip/?badge=latest
        :alt: Documentation Status

========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |travis| |appveyor|
    * - package
      - |version| |downloads| |wheel| |supported-versions| |supported-implementations|

.. |docs| image:: https://readthedocs.org/projects/pytrip/badge/?style=flat
    :target: https://readthedocs.org/projects/pytrip
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/pytrip/pytrip.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/pytrip/pytrip

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/grzanka/pytrip?branch=master&svg=true
    :alt: Appveyor Build Status
    :target: https://ci.appveyor.com/project/grzanka/pytrip

.. |version| image:: https://img.shields.io/pypi/v/pytrip98.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/pytrip98

.. |downloads| image:: https://img.shields.io/pypi/dm/pytrip98.svg?style=flat
    :alt: PyPI Package monthly downloads
    :target: https://pypi.python.org/pypi/pytrip98

.. |wheel| image:: https://img.shields.io/pypi/wheel/pytrip98.svg?style=flat
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/pytrip98

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/pytrip98.svg?style=flat
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/pytrip98

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pytrip98.svg?style=flat
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/pytrip98

.. end-badges

PyTRiP


Installation
============

Stable version ::

    pip install pytrip98

Latest unstable version, directly GIT repository, using::

    pip install setuptools versioneer
    pip install git+https://github.com/pytrip/pytrip.git

To unistall, simply use::

    pip uninstall beprof

Documentation
=============

https://pytrip.readthedocs.io/


WHAT IS THIS ?
--------------

PyTRiP is a python package for working with TRiP and VIRTUOS/VOXELPLAN files.
It is mainly supposed for batch processing, but an experimental GUI is also included.


HOW TO DO STUFF::

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



Credits
-------

This package was created with Cookiecutter_ and the `grzanka/cookiecutter-pip-docker-versioneer`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`grzanka/cookiecutter-pip-docker-versioneer`: https://github.com/grzanka/cookiecutter-pip-docker-versioneer
