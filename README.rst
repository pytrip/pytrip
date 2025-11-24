PyTRiP98
========

PyTRiP98 is a python package for working with TRiP98 and VIRTUOS/VOXELPLAN files.
It is mainly supposed for batch processing, but an experimental GUI is also included
(see https://github.com/pytrip/pytripgui).

**PyTRiP** provides several command-line applications including ``trip2dicom``, ``dicom2trip`` and ``cubeslice``.
They works under Linux, Windows and Mac OSX operating systems
(interpreter of Python programming language has to be also installed).
No programming knowledge is required from the user, but basic skills in working with the console are needed to use them.


Documentation
-------------

Full PyTRiP documentation can be found here: https://pytrip.readthedocs.io/

See `Getting Started <https://pytrip.readthedocs.org/en/stable/getting_started.html>`_ for installation and basic
information, and the `User's Guide <https://pytrip.readthedocs.org/en/stable/user_guide.html>`_ for an overview of
how to use the PyTRiP library.

Installation
------------

PyTRiP98 is distributed as pre-built wheels for CPython 3.9–3.13 on Linux, macOS and Windows. The wheels are
produced by `cibuildwheel <https://cibuildwheel.pypa.io/>`_ via GitHub Actions. NumPy 2.x is required.

Basic installation (latest release from PyPI)::

	pip install pytrip98

Optional remote execution support (adds ``paramiko``)::

	pip install pytrip98[remote]

To build from source (requires a C compiler and Python headers)::

	git clone https://github.com/pytrip/pytrip.git
	cd pytrip
	pip install .

The build is managed by ``pyproject.toml`` (PEP 621) and uses ``setuptools``; NumPy (>=2.0) headers are installed
during wheel builds to ensure a stable ABI.

Local Development
-----------------

To set up a local development environment:

1. Clone the repository::

    git clone https://github.com/pytrip/pytrip.git
    cd pytrip

2. Create and activate a virtual environment::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies and the package in editable mode::

    pip install -r requirements.txt
    pip install -r tests/requirements-test.txt
    pip install -e .

4. Run tests::

    pytest tests/

5. Build the package locally::

    pip install build
    python -m build
