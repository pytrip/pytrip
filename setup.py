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
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={"pytrip": ["data/*.dat"]},
    ext_modules=ext_modules,
)
