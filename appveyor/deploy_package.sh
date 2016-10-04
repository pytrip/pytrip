#!/usr/bin/env bash

set -x # Print command traces before executing command

set -e # Exit immediately if a simple command exits with a non-zero status.

set -o pipefail # Return value of a pipeline as the value of the last command to
                # exit with a non-zero status, or zero if all commands in the
                # pipeline exit successfully.


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
repository: https://pypi.python.org/pypi
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

echo "TAG" $APPVEYOR_REPO_TAG

ls -al dist
# upload only if tag present
if [[ $APPVEYOR_REPO_TAG == "true" ]]; then
    pip install twine
    twine upload -r pypi dist/*whl
fi

