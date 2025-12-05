.. highlight:: shell

=====================
Developer Setup Guide
=====================

This guide provides step-by-step instructions for developers who want to work on the PyTRiP98 source code.

Prerequisites
=============

Before starting, ensure you have:

* Python 3.9 or later installed (check with ``python3 --version``)
* Git installed for version control
* A C compiler (GCC on Linux, Clang on macOS, or MSVC on Windows) for building C extensions

Cloning the Repository
======================

PyTRiP98 uses Git submodules for test data. When cloning the repository, use the ``--recurse-submodules`` flag to automatically fetch the test data:

::

    git clone --recurse-submodules https://github.com/pytrip/pytrip.git

Change to the project directory:

::

    cd pytrip

If you already cloned the repository without submodules, initialize them with:

::

    git submodule update --init --recursive

Setting Up the Virtual Environment
==================================

We strongly recommend using a Python virtual environment to isolate your development dependencies.

Create a virtual environment:

::

    python3 -m venv venv

Activate the virtual environment:

On Linux or macOS:

::

    source venv/bin/activate

On Windows:

::

    venv\Scripts\activate

Installing in Editable Mode
===========================

Install PyTRiP98 in editable mode with test dependencies. This allows you to make changes to the source code and have them immediately reflected without reinstalling:

::

    pip install -e ".[test]"

If your pip complains about NumPy headers, install NumPy first:

::

    pip install numpy

Then retry:

::

    pip install -e ".[test]"

Verifying the Installation
==========================

After installation, verify that PyTRiP98 is correctly installed by starting a Python interpreter:

::

    python

Then import the module:

::

    >>> import pytrip as pt
    >>> d1 = pt.DosCube()
    >>> print("PyTRiP98 is working correctly!")

Press ``Ctrl+D`` (Linux/macOS) or ``Ctrl+Z`` followed by ``Enter`` (Windows) to exit the Python interpreter.

Running Tests
=============

Run the test suite to verify that everything is working correctly:

::

    python -m pytest tests/

To run a specific test file:

::

    python -m pytest tests/test_file_to_run.py

To run tests with verbose output:

::

    python -m pytest tests/ -v

Code Quality Checks
===================

Before submitting changes, ensure your code complies with PEP8 standards using flake8:

::

    flake8 pytrip tests

Running Command-Line Tools
==========================

Once installed in editable mode, the command-line tools are available directly:

::

    cubeslice --help

::

    trip2dicom --help

::

    dicom2trip --help

You can also run the utilities as Python modules:

::

    python -m pytrip.utils.cubeslice --help

Building Documentation
======================

To build the documentation locally, install the documentation dependencies:

::

    pip install sphinx

Then build the HTML documentation:

::

    cd docs
    sphinx-build -b html . _build/html

The built documentation will be available in ``docs/_build/html/``.

Building Distribution Packages
==============================

To build distribution artifacts (wheel and source distribution):

::

    pip install build

::

    python -m build

The built packages will be in the ``dist/`` directory.

Troubleshooting
===============

**ModuleNotFoundError: No module named 'pytrip'**

This error occurs when PyTRiP98 is not installed in your Python environment. Make sure you:

1. Have activated your virtual environment
2. Have run ``pip install -e ".[test]"``

**Import errors after updating the code**

If you encounter import errors after pulling new changes from the repository:

1. Update submodules: ``git submodule update --init --recursive``
2. Reinstall in editable mode: ``pip install -e ".[test]"``

**NumPy header errors during installation**

If you see errors about NumPy headers during the editable install:

1. Install NumPy first: ``pip install numpy``
2. Then install PyTRiP98: ``pip install -e ".[test]"``

Next Steps
==========

After setting up your development environment, see the :doc:`contributing` guide for information on how to contribute to the project, including code style guidelines and pull request procedures.
