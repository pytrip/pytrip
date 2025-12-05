"""
This setup.py is intentionally kept alongside pyproject.toml.

Why it's still needed:
- Our package ships C extensions that depend on NumPy headers and require
    build-time configuration that pyproject-only workflows don't fully cover
    for all cibuildwheel targets.
- Using setuptools + Extension here ensures consistent discovery of NumPy's
    include paths and reliable compilation across platforms.

Most metadata/configuration moved to pyproject.toml; setup.py focuses solely on
declaring and building the C extensions.
"""
from setuptools import setup, Extension, find_packages
import numpy

ext_modules = [
    Extension(
        "pytrip.pytriplib",
        sources=["pytrip/lib/core.c"],
        extra_compile_args=["-fPIC"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "pytrip._cntr",
        sources=["pytrip/lib/cntr.c"],
        extra_compile_args=["-fPIC"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="pytrip",
    packages=find_packages(exclude=[
        "tests", "tests.*",
        "docs", "docs.*",
        "examples", "examples.*",
        "build", "build.*",
        "dist", "dist.*",
    ]),
    include_package_data=True,
    package_data={"pytrip": ["data/*.dat"]},
    ext_modules=ext_modules,
)
