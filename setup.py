#
#    Copyright (C) 2010-2021 PyTRiP98 Developers.
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
"""
TODO: documentation here.
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
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        print("Building package with numpy version {}".format(numpy.__version__))
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
            return '{}+rev{}'.format(tag_name, no_of_commits_since_last_tag)
        return "Unknown"
    except OSError:
        return "Unknown"


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
    return "Unknown"


def get_version():
    version = git_version()
    if version != "Unknown":
        return version
    return pytrip_init_version()


def write_version_py(version, filename='pytrip/__init__.py'):
    with open(filename, 'a') as f:
        f.write("\n__version__ = '{:s}'".format(version))


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
# detailed list of API versions: https://github.com/numpy/numpy/blob/main/numpy/core/code_generators/cversions.txt
# we are taking the approach to build pytrip wheel package with oldest available API version for given python version
# here is table with corresponding numpy versions, numpy API version and supported python versions
# ----------------------------------------------------------------|
# | numpy version | numpy API | python versions |    OS support   |
# ----------------------------------------------------------------|
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
# |      1.11     | 10 (0xa)  | 2.7,  3.4 - 3.5 | linux, mac, win |
# |      1.10     | 10 (0xa)  | 2.7,  3.3 - 3.5 |      linux      |
# |       1.9     |  9 (0x9)  | 2.7,  3.3 - 3.5 |      linux      |
# ----------------------------------------------------------------|


install_requires = [
    "matplotlib; python_version > '3.5'",
    "matplotlib<3.1 ; python_version <= '3.5'",
    "pydicom",
    "scipy ; python_version > '3.5'",
    "scipy<1.3 ; python_version <= '3.5'",
    "kiwisolver==1.1 ; python_version <= '3.5'",
    "cffi<1.15 ; python_version <= '3.5'",
    "enum34 ; python_version < '3.5'",  # python 3.4 and 2.7
    # full range of NumPy version with support for given python version
    "numpy>=1.21.4 ; python_version == '3.10'",
    "numpy>=1.20 ; python_version == '3.9'",
    "numpy>=1.18 ; python_version == '3.8'",
    "numpy>=1.15 ; python_version == '3.7'",
    "numpy>=1.12,<1.20 ; python_version == '3.6'",
    "numpy>=1.11,<1.19 ; python_version == '3.5'",
    "numpy>=1.11,<1.15 ; python_version < '3.5'"  # python 3.4 and 2.7
]

# oldest NumPy version with support for given python version
setup_requires = [
    "numpy==1.21.4 ; python_version == '3.10'",
    "numpy==1.20.0 ; python_version == '3.9'",
    "numpy==1.18.0 ; python_version == '3.8'",
    "numpy==1.15.0 ; python_version == '3.7'",
    "numpy==1.12.0 ; python_version == '3.6'",
    "numpy==1.11.0 ; python_version < '3.6'"  # python 3.5, 3.4 and 2.7
]

extras_require = {
    'remote': ['paramiko']
}

setuptools.setup(
    name='pytrip98',
    cmdclass={'build_ext': build_ext},
    version=pytrip98_version,
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    url='https://github.com/pytrip/pytrip',
    license='GPL',
    author='Jakob Toftegaard, Niels Bassler, Leszek Grzanka',
    author_email='leszek.grzanka@ifj.edu.pl',
    description='PyTRiP',
    long_description=readme + '\n',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Physics',

        # OS and env
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: C',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython'
    ],
    package_data={'pytrip': ['data/*.dat', 'pytriplib.*', 'cntr.*']},
    setup_requires=setup_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    ext_package='pytrip',
    ext_modules=extensions,
    entry_points={
        'console_scripts': [
            'trip2dicom=pytrip.utils.trip2dicom:main',
            'dicom2trip=pytrip.utils.dicom2trip:main',
            'cubeslice=pytrip.utils.cubeslice:main',
            'rst2sobp=pytrip.utils.rst2sobp:main',
            'rst_plot=pytrip.utils.rst_plot:main',
            'bevlet2oer=pytrip.utils.bevlet2oer:main',
            'gd2dat=pytrip.utils.gd2dat:main',
            'gd2agr=pytrip.utils.gd2agr:main',
            'spc2pdf=pytrip.utils.spc2pdf:main',
        ],
    },
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.3.*')
