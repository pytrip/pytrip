#!/usr/bin/env bash

set -x # Print command traces before executing command

set -e # Exit immediately if a simple command exits with a non-zero status.

set -o pipefail # Return value of a pipeline as the value of the last command to
                # exit with a non-zero status, or zero if all commands in the
                # pipeline exit successfully.

if [[ $TOXENV == py27* ]] || [[ $TOXENV == py32* ]] ;
then
    echo "TESTS DISABLED"
else
    pyenv exec tox --notest -e $TOXENV
    pyenv exec tox -- -n 1
fi

