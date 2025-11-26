.. highlight:: shell

Detailed Installation Guide
===========================
This guide covers prerequisites, wheel installation from PyPI and building from source using ``pyproject.toml``.


Prerequisites
-------------

**PyTRiP** works under Linux and Mac OSX operating systems.

First we need to check if Python interpreter is installed.
Try if one of following commands (printing Python version) works::

    $ python --version
    $ python3 --version

**pytrip** supports modern Python versions 3.9 – 3.14. Ensure your interpreter is within this range.

If none of ``python`` and ``python3`` commands are present, then Python interpreter has to be installed.

We suggest to use the newest version available (from 3.x family).


Python installers can be found at the python web site
(http://python.org/download/).

PyTRiP relies on these core packages which are installed automatically when using pip:

    * `NumPy <http://www.numpy.org/>`_ – Array and numerical operations (ABI handled during wheel builds).
    * `SciPy <https://scipy.org/>`_ – Scientific routines.
    * `matplotlib <http://matplotlib.org/>`_ – Plotting.
    * `pydicom <https://pydicom.github.io/>`_ – DICOM handling.
    * `packaging <https://pypi.org/project/packaging/>`_ – Version parsing.

Optional remote execution support via SSH requires:

    * `paramiko <http://www.paramiko.org/>`_ – Install with ``pip install pytrip98[remote]``.

Installing from PyPI (All Platforms)
-----------------------------------

The easiest way to install PyTRiP98 is using `pip <https://pypi.python.org/pypi/pip>`_ wheels. Pre-built wheels
are generated for Linux, macOS and Windows using `cibuildwheel <https://cibuildwheel.pypa.io/>`_ and include the
compiled C extensions.

Basic installation::

    pip install pytrip98

With remote execution extras::

    pip install pytrip98[remote]


Administrator (system-wide) install::

    sudo pip install pytrip98

Upgrade system-wide installation::

    sudo pip install --upgrade pytrip98

Uninstall::

    sudo pip uninstall pytrip98


User (local) install::

    pip install pytrip98 --user

Upgrade local installation::

    pip install --upgrade pytrip98 --user

Uninstall::

    pip uninstall pytrip98

Executables (e.g. ``cubeslice``) will be placed in ``$HOME/.local/bin``; ensure this directory is on your ``PATH``.

Building From Source
--------------------

If you need to build from source (e.g. for development or unsupported architectures)::

    git clone https://github.com/pytrip/pytrip.git
    cd pytrip
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -e .

During wheel builds a minimal NumPy version appropriate for your Python interpreter is pre-installed to ensure
stable binary compatibility of the C extensions.

