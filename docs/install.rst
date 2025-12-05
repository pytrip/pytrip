.. highlight:: shell

Detailed Installation Guide
===========================
This guide covers prerequisites, virtual environments, wheel installation from PyPI, and building from source using ``pyproject.toml``.


Prerequisites
-------------

**PyTRiP** works under Linux, macOS, and Windows.

For contributors and developers, we strongly recommend working on Linux or macOS for the smoothest build and tooling experience.

First we need to check if Python interpreter is installed.
Try if one of following commands (printing Python version) works::

    $ python --version
    $ python3 --version

**pytrip** supports modern Python versions 3.9 – 3.14. Ensure your interpreter is within this range.

If none of ``python`` and ``python3`` commands are present, then Python interpreter has to be installed.

We suggest to use the newest version available (from 3.x family).


Python installers can be found at the python web site
(http://python.org/download/).

PyTRiP relies on these core packages which are installed automatically when using pip or editable installs:

    * `NumPy <http://www.numpy.org/>`_ – Array and numerical operations (ABI handled during wheel builds).
    * `SciPy <https://scipy.org/>`_ – Scientific routines.
    * `matplotlib <http://matplotlib.org/>`_ – Plotting.
    * `pydicom <https://pydicom.github.io/>`_ – DICOM handling.
    * `packaging <https://pypi.org/project/packaging/>`_ – Version parsing.

Optional remote execution support via SSH requires:

    * `paramiko <http://www.paramiko.org/>`_ – Install with ``pip install "pytrip98[remote]"``.

Installing from PyPI (All Platforms)
------------------------------------

The easiest way to install PyTRiP98 is using `pip <https://pypi.python.org/pypi/pip>`_ wheels. Pre-built wheels
are generated for Linux, macOS and Windows using `cibuildwheel <https://cibuildwheel.pypa.io/>`_ and include the
compiled C extensions.

We strongly recommend using a virtual environment:

Create and activate a virtual environment::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the latest release from PyPI::

    pip install pytrip98

With remote execution extras::

    pip install "pytrip98[remote]"

Executables (e.g. ``cubeslice``) will be placed in your virtual environment's ``bin`` (Linux/macOS) or ``Scripts`` (Windows) directory.

Installing from git
-------------------

To install pytrip98 directly from git in a virtual environment, use the following commands.

Create and activate a virtual environment::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Install from the master branch::

    pip install git+https://github.com/pytrip/pytrip.git@master

Install from a specific branch (e.g., a feature branch)::

    pip install git+https://github.com/pytrip/pytrip.git@branch-name

To install with additional dependencies, use the extras syntax::

    pip install "git+https://github.com/pytrip/pytrip.git@master#egg=pytrip98[remote]"

Building From Source
--------------------

If you need to build from source (e.g. for development or unsupported architectures)::

    git clone --recurse-submodules https://github.com/pytrip/pytrip.git
    cd pytrip
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    # Editable install with test tools (pytest/flake8)
    pip install -e .[test]
    # If your pip complains about NumPy headers, install NumPy first:
    #   pip install numpy

Run the test suite to verify your setup::

    python -m pytest tests/

Optionally build distribution artifacts (wheel and sdist)::

    pip install build
    python -m build

During official wheel builds a minimal NumPy version appropriate for your Python interpreter is pre-installed to ensure
stable binary compatibility of the C extensions.

