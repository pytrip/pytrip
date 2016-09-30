.. highlight:: shell

Installation guide
==================

**pytrip** works under Linux and Mac OSX operating systems.

If you are familiar with python and pip tool, simply type following command to install the package::

    $ sudo pip install pytrip98

If your are a less advanced user, read the rest of the page.

Installation guide is divided in two phases: prerequisites (mainly Python interpreter installation) and main package installation.

There are two groups of users: administrators and regular ones.
For regular users we assume that they can write files to their home directory, but not necessary elsewhere.
We assume that administrators can log in as root user, execute commands via ``sudo`` or
have some other way to create files system-wide. This instruction is aimed at these two groups.


.. contents:: Table of Contents
   :depth: 2

Prerequisites - python interpreter
----------------------------------

First we need to check if Python interpreter is installed.
Try if one of following commands (printing Python version) works::

    $ python --version
    $ python3 --version

At the time of writing Python language interpreter has two popular versions: 2.x (Python 2) and 3.x (Python 3) families.
Command ``python`` invokes either Python 2 or 3, while ``python3`` can invoke only Python 3.

**pytrip** supports most of the modern Python versions, mainly: 2.7, 3.2, 3.3, 3.4, 3.5 and 3.6.
Check if your interpreter version is supported.

If none of ``python`` and ``python3`` commands are present, then Python interpreter has to be installed.

We suggest to use the newest version available (from 3.x family).

Administrator installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The best way is to use your package manager.

* ``sudo apt-get install python3`` (python 3) or ``sudo apt-get install python`` (python 2) for Debian and Ubuntu
* ``sudo dnf install python3`` (python 3) or ``sudo dnf install python`` (python 2) for Fedora
* ``sudo yum install python3`` (python 3) or ``sudo yum install python`` (python 2) for CentOS and SLC
* ``sudo pacman -S python`` (python 3) or ``sudo pacman -S python2`` (python 2) for archLinux
* ``sudo zypper install python3`` (python 3) or ``sudo zypper install python`` (python 2) for openSUSE


User installation
~~~~~~~~~~~~~~~~~

In case Python is missing and you are regular user, the best would be contact somebody with administrator
rights and ask to install Python interpreter in the system.

Installation of Python without administrator rights is however possible, but in case something goes
wrong it will require expert knowledge.

As an user you will need to download Python interpreter source code (written in C language) and compile it.
For that purpose you will need a C language compiler (i.e. `gcc`) and some other tools (i.e. `make`).
These tools are usually installed by somebody with administrator rights.
Python installer might not complain about missing SSL libraries (i.e. `libssl-dev`) and will compile successfully,
but we recommend to install it (SSL libraries) before,
to have easier installation of pip package manager in the next steps.

When installing as user we advice to unpack downloaded source code in `$HOME/tmp` directory and keep it there.
It may be needed for upgrade or deinstallation purpose.

Let us install Python 2.7 into `$HOME/usr/py27` directory. First let us create `$HOME/tmp` directory and step into it::

    $ mkdir -p $HOME/tmp
    $ cd $HOME/tmp

Now its time to download and unpack source code package. We show an example with 2.7.12 version, but newer one can
be problably found on https://www.python.org/downloads/source/ ::

    $ wget https://www.python.org/ftp/python/2.7.12/Python-2.7.12.tgz
    $ tar -zxf Python-2.7.12.tgz
    $ cd Python-2.7.12

Finally let us start compilation process (it might take couple of minutes). This process might be interrupted by
some error message. Do not hesitate to find a professional help to fix it::

    $ ./configure --prefix=$HOME/usr/py27
    $ make
    $ make install

Python2.7 is now installed into `$HOME/usr/py27` directory.
In order to execute python interpreter, you need to provide full path to the executable file, i.e.::

    $ $HOME/usr/py27/bin/python --version

In a similar way python3.x can be installed.


Prerequisites - pip tool
------------------------

**pip** is a tool for installing and managing Python packages.
It downloads the packages from central Internet repository and installs them
in a similar way as apps are downloaded on your smartphone by Google Play or Apple Store.

Try the following commands (printing pip version)::

    $ pip --version
    $ pip3 --version

In a similar way to python interpreter pip is a tool for Python 2 or 3,
while pip3 works exclusively for Python 3.
If none of these commands are present, then pip has to be installed.


Administrator installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the package installation for your system.
On some systems instructions mentioned below have to be prefixed with `sudo` command.

* ``sudo apt-get install python3-pip`` (python 3) or ``sudo apt-get install python-pip`` (python 2) for Debian and Ubuntu
* ``sudo dnf install python3-pip`` (python 3) or ``sudo dnf install python-pip`` (python 2) for Fedora
* ``sudo yum install python3-pip`` (python 3) or ``sudo yum install python-pip`` (python 2) for CentOS and SLC
* ``sudo pacman -S python-pip`` (python 3) or ``sudo pacman -S install python2-pip`` (python 2) for archLinux
* ``sudo zypper install python3-pip`` (python 3) or ``sudo zypper install python-pip`` (python 2) for openSUSE


User installation
~~~~~~~~~~~~~~~~~

Follow the instruction from here https://pip.pypa.io/en/stable/installing/,
mainly - download installation script using your web browser, or by typing in the terminal::

    $ wget https://bootstrap.pypa.io/get-pip.py

# openssl for python2

Now use your python interpreter to execute downloaded script. It will install pip in your home directory::

    $ python get-pip.py --user

Try if pip command is available by typing::

    $ $HOME/.local/bin/pip --version

If this method fails you can also try to use a `ensurepip` approach.
It works with Python versions: 2.7 (starting from 2.7.9), 3.4 and newer.
To install pip, simply type::

    $ python -m ensurepip


pytrip pip package installation
-------------------------------

Now it is time to install **pytrip** package.


Administrator installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Administrator installation is very simple, but requires to save some files in system-wide directories (i.e. `/usr`)::

    $ sudo pip install pytrip98

To upgrade the **pytrip** to newer version, simply type::

    $ sudo pip install --upgrade pytrip98

To completely remove **pytrip** from your system, use following command::

    $ sudo pip uninstall pytrip98

Now all **pytrip** commands should be installed for all users::

    $ cubeslice --help


User installation
~~~~~~~~~~~~~~~~~

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

