.. highlight:: shell

Detailed Installation Guide
===========================
Installation guide is divided in two phases: checking the prerequisites and main package installation.


Prerequisites
-------------

**PyTRiP** works under Linux and Mac OSX operating systems.

First we need to check if Python interpreter is installed.
Try if one of following commands (printing Python version) works::

    $ python --version
    $ python3 --version

At the time of writing Python language interpreter has two popular versions: 2.x (Python 2) and 3.x (Python 3) families.
Command ``python`` invokes either Python 2 or 3, while ``python3`` can invoke only Python 3.

**pytrip** supports most of the modern Python versions, mainly: 2.7, 3.5 - 3.10.
Check if your interpreter version is supported.

If none of ``python`` and ``python3`` commands are present, then Python interpreter has to be installed.

We suggest to use the newest version available (from 3.x family).


Python installers can be found at the python web site
(http://python.org/download/).

PyTRiP also relies on these packages:

  * `NumPy <http://www.numpy.org/>`_ -- Better arrays and data processing.
  * `matplotlib <http://matplotlib.org/>`_ -- Needed for plotting.
  * `paramiko <http://www.paramiko.org/>`_ -- Needed for remote execution of TRiP98 via SSH.

and if they are not installed beforehand, these will automatically be fetched by pip.

Installing using pip (all platforms)
------------------------------------

The easiest way to install PyTRiP98 is using `pip <https://pypi.python.org/pypi/pip>`_::

.. note::
    Pip comes pre-installed with Python newer than 3.4 and 2.7 (for 2.x family)


Administrator installation (root access)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Administrator installation is very simple, but requires to save some files in system-wide directories (i.e. `/usr`)::

    $ sudo pip install pytrip98

To upgrade the **pytrip** to newer version, simply type::

    $ sudo pip install --upgrade pytrip98

To completely remove **pytrip** from your system, use following command::

    $ sudo pip uninstall pytrip98

Now all **pytrip** commands should be installed for all users::

    $ cubeslice --help


User installation (non-root access)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User installation will put the **pytrip** under hidden directory `$HOME/.local`.

To install the package, type in the terminal::

    $ pip install pytrip98 --user

If `pip` command is missing on your system, replace `pip` with `pip3` in abovementioned instruction.

To upgrade the **pytrip** to newer version, simply type::

    $ pip install --upgrade pytrip98 --user

To completely remove **pytrip** from your system, use following command::

    $ pip uninstall pytrip98

In most of modern systems all executables found in `$HOME/.local/bin` directory can be called
like normal commands (i.e. `ls`, `cd`). It means that after installation you should be able
to simply type in terminal::

    $ cubeslice --help

If this is not the case, please prefix the command with `$HOME/.local/bin` and call it in the following way::

    $ $HOME/.local/bin/cubeslice --help

