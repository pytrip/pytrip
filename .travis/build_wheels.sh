#!/bin/bash

# script based on https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh
set -e -x

# freetype v2.3.0 installation, source versions of matplotlib require it to get installed via pip
# docker image has only freetype v2.2
install_freetype() {
    sudo apt-get install --no-install-recommends -y wget
    wget --no-check-certificate http://downloads.sourceforge.net/freetype/freetype-2.3.0.tar.gz
    tar -zxvf freetype-2.3.0.tar.gz
    cd freetype-2.3.0
    ./configure --prefix=/usr
    make -j4
    make install
    cd ..
}

# python versions to be used
PYVERS=$1

# tk-devel is needed to run code dependent on matplotlib
yum install -y tk-devel

# loop over python targets and make pytrip wheel package for each of it
for TARGET in $PYVERS;
do
    # Set python location
    PYBIN=/opt/python/$TARGET/bin

    # Check pip version
    ${PYBIN}/pip -V

    # List installed packages
    ${PYBIN}/pip freeze

    # Install versioneer and generate versioneer stuff
    ${PYBIN}/pip install versioneer
    cd /io
    ${PYBIN}/versioneer install
    cd -

    # see https://github.com/google/python-subprocess32/issues/12#issuecomment-337806379
    # installation of subprocess32 port package from source fails, so let us use
    # current development version (--pre enables that)
    ${PYBIN}/pip install --pre "subprocess32 ; python_version < '3.0'"

    # Install requirements and get the exit code
    # do not include pre-releases (i.e. RC - release candidates) and development versions
    # forcing use of binary package for matplotlib as source package has API confict with numpy
    # see PR #419
    set +e
    ${PYBIN}/pip install --upgrade -r /io/requirements.txt --only-binary matplotlib
    RET_CODE=$?
    set -e

    # is normal installation failed, try install freetype and try again,
    # this time with source package for matplotlib
    if [ $RET_CODE -ne 0 ]; then
        install_freetype
        # libpng is needed by matplotlib, blas and lapack by numpy
        yum install -y libpng-devel lapack-devel blas-devel atlas-devel
        ${PYBIN}/pip install --upgrade -r /io/requirements.txt
    fi

    # Make a wheel
    ${PYBIN}/pip wheel --no-deps /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/pytrip98*.whl; do
   auditwheel show $whl
   auditwheel repair $whl -w /io/wheelhouse/
done


for TARGET in $PYVERS;
do
    # Set python location
    PYBIN=/opt/python/$TARGET/bin

    # Test if generated wheel can be installed
    # ignore package index (and pypi server), looking at local directory
    ${PYBIN}/pip install pytrip98 --no-index --find-links /io/wheelhouse

    ${PYBIN}/pip freeze
    ${PYBIN}/pip show pytrip98
    ls -al ${PYBIN}

    ## tests could fail
    ${PYBIN}/dicom2trip --help
    ${PYBIN}/trip2dicom --help
    ${PYBIN}/cubeslice --help
    ${PYBIN}/rst2sobp --help
    ${PYBIN}/rst_plot --help
    ${PYBIN}/gd2dat --help
    ${PYBIN}/gd2agr --help
    ${PYBIN}/bevlet2oer --help
    ${PYBIN}/dicom2trip --version
    ${PYBIN}/trip2dicom --version
    ${PYBIN}/cubeslice --version
    ${PYBIN}/rst2sobp --version
    ${PYBIN}/gd2dat --version
    ${PYBIN}/gd2agr --version
    ${PYBIN}/rst_plot --version
    ${PYBIN}/bevlet2oer --version
done
