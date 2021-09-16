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
import sys
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
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'describe', '--tags', '--long'])
        GIT_REVISION = out.strip().decode('ascii')
        print('GIT_REVISION', GIT_REVISION)
        if GIT_REVISION:
            no_of_commits_since_last_tag = int(GIT_REVISION.split('-')[1])
            tag_name = GIT_REVISION.split('-')[0][1:]
            if no_of_commits_since_last_tag == 0:
                version = tag_name
            else:
                version = '{}+rev{}'.format(tag_name, no_of_commits_since_last_tag)
        else:
            version = "Unknown"
    except OSError:
        version = "Unknown"

    return version


def write_version_py(filename='pytrip/__init__.py'):
    cnt = """
__version__ = '%(version)s'
"""

    GIT_REVISION = git_version()
    a = open(filename, 'a')
    try:
        a.write(cnt % {'version': GIT_REVISION})
    finally:
        a.close()


write_version_py()

with open('README.rst') as readme_file:
    readme = readme_file.read()

extensions = [
    setuptools.Extension('pytriplib', sources=[os.path.join('pytrip', 'lib', 'core.c')], extra_compile_args=['-fpic']),
    setuptools.Extension('_cntr', sources=[os.path.join('pytrip', 'lib', 'cntr.c')], extra_compile_args=['-fpic'])
]

install_requires = ["matplotlib", "pydicom"]

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
setup_requires = []
if sys.version_info[0] == 3 and sys.version_info[1] == 9:  # python 3.9
    setup_requires += ["numpy==1.20.0"]  # numpy 1.20, API v14 (0xe)
    install_requires += ["numpy>=1.20.0"]  # numpy 1.20 or newer, API v14 (0xe)
elif sys.version_info[0] == 3 and sys.version_info[1] == 8:  # python 3.8
    setup_requires += ["numpy==1.18.0"]  # numpy 1.18, API v13 (0xd)
    install_requires += ["numpy>=1.18.0"]  # numpy 1.18 or newer, API v13 (0xd)
elif sys.version_info[0] == 3 and sys.version_info[1] == 7:  # python 3.7
    setup_requires += ["numpy==1.15.0"]  # numpy 1.15, API v12 (0xc)
    install_requires += ["numpy>=1.15.0"]  # numpy 1.15 or newer, API v12 (0xc)
elif sys.version_info[0] == 3 and sys.version_info[1] == 6:  # python 3.6
    setup_requires += ["numpy==1.12.0"]  # numpy 1.12, API v10 (0xa)
    install_requires += ["numpy>=1.12.0,<1.20"]  # numpy 1.12 - 1.19, API v10 (0xa)
elif sys.version_info[0] == 3 and sys.version_info[1] == 5:  # python 3.5
    setup_requires += ["numpy==1.11.0"]  # numpy 1.11, API v10 (0xa)
    install_requires += ["numpy>=1.11.0,<1.19"]  # numpy 1.11 - 1.18, API v10 (0xa)
elif (sys.version_info[0] == 3 and sys.version_info[1] < 5) or (sys.version_info[0] == 2):  # python 3.4 + 2.7
    setup_requires += ["numpy==1.11.0"]  # numpy 1.11, API v10 (0xa)
    install_requires += ["numpy>=1.11.0,<1.15"]  # numpy 1.11 - 1.15, API v10 (0xa)
else:  # others
    setup_requires += ["numpy"]  # try newest version, this will probably fail
    install_requires += ["numpy"]  # try newest version, this will probably fail

setuptools.setup(
    name='pytrip98',
    cmdclass={'build_ext': build_ext},
    version=git_version(),
    packages=setuptools.find_packages(exclude=["tests"]),
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
        'Programming Language :: Python :: Implementation :: CPython'
    ],
    package_data={'pytrip': ['data/*.dat', 'pytriplib.*', 'cntr.*']},
    setup_requires=setup_requires,
    install_requires=install_requires,
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
