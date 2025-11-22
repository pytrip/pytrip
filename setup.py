#
#    Copyright (C) 2010-2023 PyTRiP98 Developers.
#
#    This file is part of PyTRiP98.
#
#    PyTRiP98 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyTRiP98 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyTRiP98.  If not, see <http://www.gnu.org/licenses/>.
#
"""Legacy setup.py retained for compatibility.

Primary build configuration has migrated to ``pyproject.toml``. This script
still supplies dynamic version stamping for environments invoking ``setup.py``
directly, but new workflows (CI, wheel builds) should rely on ``pyproject.toml``.
"""
import os
import setuptools
import subprocess
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    """
    From https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py/21621689#21621689
    """
    def finalize_options(self):
        _build_ext.finalize_options(self)

        import numpy

        # Prevent numpy from thinking it is still in its setup process
        # http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
        #
        # Newer numpy versions don't support this hack, nor do they need it.
        # https://github.com/pyvista/pyacvd/pull/23#issue-1298467701
        #
        # inspired by https://github.com/piskvorky/gensim/commit/2fd3e89ca42a7812a71c608572aba2e858377c8c
        import builtins
        try:
            builtins.__NUMPY_SETUP__ = False
        except Exception as ex:
            print(f'could not use __NUMPY_SETUP__ hack (numpy version: {numpy.__version__}): {ex}')

        print("Building package with numpy version {}".format(numpy.__version__))  # skipcq: PYL-C0209
        self.include_dirs.append(numpy.get_include())


def git_version():
    """
    Inspired by https://github.com/numpy/numpy/blob/master/setup.py
    :return: the git revision as a string
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        FNULL = open(os.devnull, 'w')
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=FNULL, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'describe', '--tags', '--long'])
        GIT_REVISION = out.strip().decode('ascii')
        if GIT_REVISION:
            no_of_commits_since_last_tag = int(GIT_REVISION.split('-')[1])
            tag_name = GIT_REVISION.split('-')[0][1:]
            if no_of_commits_since_last_tag == 0:
                return tag_name
            return '{}+rev{}'.format(tag_name, no_of_commits_since_last_tag)  # skipcq: PYL-C0209
        return "0.0.0"
    except OSError:
        return "0.0.0"


def pytrip_init_version():
    """
    read pytrip/__init__.py file and get __version__ variable
    we don't import it, because that module may require packages that are not available yet
    :return: version from pytrip
    """
    with open("pytrip/__init__.py", "r") as f:
        lines = f.readlines()
    for line in reversed(lines):
        if line.startswith("__version__"):
            line = line.split('#')[0]  # remove comment
            delim = '"' if '"' in line else "'"  # check if string is in " or '
            version = line.split(delim)[1]
            return version
    return "0.0.0"


def get_version():
    version = git_version()
    if version != "0.0.0":
        return version
    return pytrip_init_version()


def write_version_py(version, filename='pytrip/__init__.py'):
    if not filename.endswith('.py'):
        print("Wrong filename")
    with open(filename, 'a') as f:
        f.write("\n__version__ = '{:s}'".format(version))  # skipcq: PYL-C0209


pytrip98_version = get_version()
write_version_py(pytrip98_version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

extensions = [
    setuptools.Extension('pytriplib', sources=[os.path.join('pytrip', 'lib', 'core.c')], extra_compile_args=['-fpic']),
    setuptools.Extension('_cntr', sources=[os.path.join('pytrip', 'lib', 'cntr.c')], extra_compile_args=['-fpic'])
]


# packages specified in setup_requires are needed only when running setup.py, in our case it is only numpy
# which needs to provide header files (via numpy.get_include()) required to build C extension
# numpy is also added install_requires which is list of dependencies needed by pip when running `pip install`
#
# from time to time numpy is introducing new binary API
# detailed list of API versions: https://github.com/numpy/numpy/blob/main/numpy/_core/code_generators/cversions.txt
# we are taking the approach to build pytrip wheel package with oldest available API version for given python version
# here is table with corresponding numpy versions, numpy API version and supported python versions
# ----------------------------------------------------------------|
# | numpy version | numpy API | python versions |    OS support   |
# ----------------------------------------------------------------|
# |    2.1.0-     | 19 (0x13) |   3.10 - 3.13   | linux, mac, win |
# |    1.26.0-    | 18 (0x12) |   3.9 - 3.12    | linux, mac, win |
# |    1.25.0-    | 17 (0x11) |   3.9 - 3.11    | linux, mac, win |
# |    1.23.3-    | 16 (0x10) |    3.8 - 3.11   | linux, mac, win |
# | 1.23.0-1.23.2 | 16 (0x10) |    3.8 - 3.10   | linux, mac, win |
# |      1.22     | 15 (0xf)  |    3.8 - 3.10   | linux, mac, win |
# |   1.21.4-     | 14 (0xe)  |    3.7 - 3.10   | linux, mac, win |
# | 1.21.0-1.21.3 | 14 (0xe)  |    3.7 - 3.9    | linux, mac, win |
# |      1.20     | 14 (0xe)  |    3.7 - 3.9    | linux, mac, win |
# |      1.19     | 13 (0xd)  |    3.6 - 3.8    | linux, mac, win |
# |      1.18     | 13 (0xd)  |    3.5 - 3.8    | linux, mac, win |
# |      1.17     | 13 (0xd)  |    3.5 - 3.7    | linux, mac, win |
# |      1.16     | 13 (0xd)  | 2.7,  3.5 - 3.7 | linux, mac, win |
# |      1.15     | 12 (0xc)  | 2.7,  3.4 - 3.7 | linux, mac, win |
# |      1.14     | 12 (0xc)  | 2.7,  3.4 - 3.6 | linux, mac, win |
# |      1.13     | 11 (0xb)  | 2.7,  3.4 - 3.6 | linux, mac, win |
# |      1.12     | 10 (0xa)  | 2.7,  3.4 - 3.6 | linux, mac, win |
# ----------------------------------------------------------------|


install_requires = [
    "matplotlib",
    "pydicom",
    "scipy",
    "packaging",
    # full range of NumPy version with support for given python version
    "numpy>=2.1.0 ; python_version == '3.13'",
    "numpy>=1.26.0,<2.0 ; python_version == '3.12'",
    "numpy>=1.23.3,<2.0 ; python_version == '3.11'",
    "numpy>=1.21.4,<2.0 ; python_version == '3.10'",
    "numpy>=1.20,<2.0 ; python_version == '3.9'"
]

# oldest NumPy version with support for given python version
setup_requires = [
    "numpy==2.1.0 ; python_version == '3.13'",
    "numpy==1.26.0 ; python_version == '3.12'",
    "numpy==1.23.3 ; python_version == '3.11'",
    "numpy==1.21.4 ; python_version == '3.10'",
    "numpy==1.20.0 ; python_version == '3.9'"
]

extras_require = {
    'remote': ['paramiko']
}

if __name__ == "__main__":
    # Minimal invocation for legacy environments.
    setuptools.setup(
        name='pytrip98',
        cmdclass={'build_ext': build_ext},
        version=pytrip98_version,
        packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
        ext_package='pytrip',
        ext_modules=extensions,
        install_requires=install_requires,
    )
