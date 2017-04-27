#!/usr/bin/env bash

set -x # Print command traces before executing command

set -e # Exit immediately if a simple command exits with a non-zero status.

set -o pipefail # Return value of a pipeline as the value of the last command to
                # exit with a non-zero status, or zero if all commands in the
                # pipeline exit successfully.

# generate versioneer files: versioneer.py, pytrip/_version.py
# and modify accordingly MANIFEST.in and pytrip/__init__.py 
versioneer install

# keeping generated files in repo is bad practice
# versioneer.py, pytrip/_version.py and modifications in MANIFEST.in and pytrip/__init__.py 
# are exluded from staging by following commands: 
git reset HEAD versioneer.py
git reset HEAD pytrip/_version.py
git reset HEAD MANIFEST.in
git reset HEAD pytrip/__init__.py
git reset HEAD .gitattributes
