.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/pytrip/pytrip/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs or Implement Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for bugs or features.
Anything tagged with "bug" or "feature" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "feature"
is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

`pytrip` could always use more documentation, whether as part of the
official `pytrip` docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/pytrip/pytrip/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started for GIT-aware developers
------------------------------------

Ready to contribute? Here's how to set up `pytrip` for local development.
We assume you are familiar with GIT source control system. If not you will
other instruction at the end of this page.

1. Fork the `pytrip` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/pytrip.git

3. If you are not familiar with GIT, proceed to step 5, otherwise create a branch for local development::

    $ cd pytrip
    $ git checkout -b feature/issue_number-name_of_your_bugfix_or_feature

4. Now you can make your changes locally.

As the software is prepared to be shipped as pip package, some modifications
of PYTHONPATH variables are needed to run the code. Let us assume you are now in the same directory as ``setup.py`` file.


The standard way to execute Python scripts WILL NOT WORK::

   $ python pytrip/utils/cubeslice.py --help

It will probably give you a traceback like this one::

    Traceback (most recent call last):
    File ".\pytrip\utils\cubeslice.py", line 29, in <module>
        import pytrip as pt
    ModuleNotFoundError: No module named 'pytrip'

To have the code working (as a developer), you need to call the files as python modules.
In this way python interpreter will set properly all directories needed for proper imports::

   $ python -m pytrip.utils.cubeslice --help
   usage: cubeslice.py [-h] [--data [DATA]] [--ct [CT]] [-v] [-f N] [-t M] [-H]
                    [-o OUTPUTDIR]

   (...)

5. Make local changes to fix the bug or to implement a feature.

6. When you're done making changes, check that your changes comply with PEP8 code quality standards (flake8 tests) run pytest tests::

    $ pep8 --max-line-length=120 pytrip tests
    $ flake8 --max-line-length=120 pytrip tests
    $ pytest

   To get pep8, flake8 and pytest, just pip install them.

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."

8. Repeat points 4-6 until the work is done. Now its time to push the changes to remote repository::

    $ git push origin feature/issue_number-name_of_your_bugfix_or_feature

9. Submit a pull request through the GitHub website to the master branch of ``git@github.com:pytrip/pytrip.git`` repository.

10. Check the status of automatic tests

You can find them on the pull request webpage https://github.com/pytrip/pytrip/pulls.
In case some of the tests fails, fix the problem. Then commit and push your changes (steps 5-8).


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 2.7, 3.5-3.10. Check
   https://github.com/pytrip/pytrip/actions
   and make sure that the tests pass for all supported Python versions.


Get Started for non-GIT developers
----------------------------------

1. Fetch the code from remote GIT repository to your local directory::

    $ git clone git@github.com:pytrip/pytrip.git

2. Follow steps 4-6 from the instruction for GIT-aware developers. To run code locally, prefix usual calls with ``PYTHONPATH=.``::

   $ python -m pytrip.utils.cubeslice --help
   usage: cubeslice.py [-h] [--data [DATA]] [--ct [CT]] [-v] [-f N] [-t M] [-H]
                    [-o OUTPUTDIR]

   (...)

Make your changes and check that they comply with PEP8 code quality standards (flake8 tests) and run pytest::

    $ flake8 pytrip tests
    $ pytest

3. Compress your working directory and send it to us by email (see `authors <AUTHORS.rst>`__), describing your changes.


Tips
----

To run full tests type::

    $ pytest

To run only a single test type::

   $ python -m pytest tests/test_file_to_run.py

.. _`bugs`: https://github.com/pytrip/pytrip/issues
.. _`features`: https://github.com/pytrip/pytrip/issues


Nomenclature
------------

1. Classes: CamelTyped. Example: ``class CtxCube()``
2. Methods and functions: lowercase , typically containing a verb and separated by underscore. Example: ``def save_cube()``
    * avoid ``get_*`` and ``set_*`` functions as this is not pythonic.
3. Attributes and variables: lowercase and typically consisting of one or more nouns separated by underscore. Example: ``self.target_dose``
4. Functions, class methods, attributes etc which are not supposed to be accessed by users should be prefixed with underscore i.e. ``_foobar``
5. Directories, paths and filenames should be named following this scheme:

* **Filenames**
    * ``funk.dat`` : filename
    * ``funk`` : basename
* **Directories**
    * ``/home/bassler/foobar`` : absolute directory ``abs_dir``
    * ``./foobar`` : relative directory ``rel_dir``
    * or just ``dir`` if both may be applicable.
* **Paths**
    * /home/bassler/foobar/funk.dat : absolute ``abs_path``
    * ``foobar/funk.dat`` : (relative) path ``rel_path``
    * prefix ``path`` with ``root_`` if it is without file extension.
    * ``/home/bassler/foobar/funk`` : root path ``root_path``
    * ``./foobar/funk`` : root path ``root_path``
    * or just ``path`` if any may be applicable.

* More details on attribute name **prefixes**:
    * ``abs_`` -> absolute path to file or directory, starting with ``/`` or ``C:\`` (see ``os.path.abspath``)
    * ``root_`` -> root part of path (may be absolute or relative, see ``os.path.splitext``)
    * ``rel_`` -> relative path (see ``os.path.relpath``)
