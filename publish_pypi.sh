#!/bin/bash
# Script to build and publish python package to PyPI

# Ensure twine and build are installed
pip install twine build

rm -rf dist

# Build the package
python -m build

# Publish to PyPI
# Note: Requires ~/.pypirc or interactive login
python -m twine upload dist/*
