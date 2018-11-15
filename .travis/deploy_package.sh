#!/usr/bin/env bash

set -x # Print command traces before executing command

set -e # Exit immediately if a simple command exits with a non-zero status.

set -o pipefail # Return value of a pipeline as the value of the last command to
                # exit with a non-zero status, or zero if all commands in the
                # pipeline exit successfully.

# python target to be used (cp27-cp27m, cp27-cp27mu, cp34-cp34m or cp35-cp35m)
TARGET=$1

# optional pre-command (i.e. linux32)
PRE_CMD=$2

# based on https://github.com/pypa/python-manylinux-demo/blob/master/.travis.yml
DOCKER_IMAGE_32=quay.io/pypa/manylinux1_i686
DOCKER_IMAGE_64=quay.io/pypa/manylinux1_x86_64

write_pypirc() {
PYPIRC=~/.pypirc

if [ -e "${PYPIRC}" ]; then
    rm ${PYPIRC}
fi

touch ${PYPIRC}
cat <<pypirc >${PYPIRC}
[distutils]
index-servers =
    pypi

[pypi]
username: ${PYPIUSER}
password: ${PYPIPASS}

pypirc

if [ ! -e "${PYPIRC}" ]; then
    echo "ERROR: Unable to write file ~/.pypirc"
    exit 1
fi
}

# write .pypirc file with pypi repository credentials
set +x
write_pypirc
set -x

echo "User" $PYPIUSER

# make a source package
if [[ $TARGET == "source" ]]; then
    # install necessary tools
    pip install -U wheel twine

    # makes source
    python setup.py sdist

    ls -al dist
    # upload only if tag present
    if [[ $TRAVIS_TAG != "" ]]; then
        twine upload dist/*tar.gz
    fi
else

# Building of manylinux1 compatible packages, see https://www.python.org/dev/peps/pep-0513/ for details
# pytrip98 has C extension, pip repo accepts only packages tagged as manylinux1
# to get manylinux1 package, you are forced to use some ancient Linux version (here CentOS 5)
# in such case C extension will be linked against old glibc thus granting maximum portability
    if [[ $PRE_CMD == "linux32" ]]; then
        DOCKER_IMAGE=$DOCKER_IMAGE_32
    else
        DOCKER_IMAGE=$DOCKER_IMAGE_64
    fi
    docker pull $DOCKER_IMAGE

    # run building script
    docker run --rm -v `pwd`:/io $DOCKER_IMAGE $PRE_CMD /io/.travis/build_wheels.sh "$TARGET"

    ls -al wheelhouse/ # just for debugging
    # install necessary tools
    pip install -U wheel twine

    # upload only if tag present
    if [[ $TRAVIS_TAG != "" ]]; then
        twine upload wheelhouse/*
    fi
fi
