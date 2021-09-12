#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./make_pypy_release.sh X.Y.Z"
    exit 2
fi

# Make release
cd ..
git tag $1 -m "pycollector-$1"
git push --tags origin master

python3 setup.py sdist bdist_wheel
twine upload dist/*

rm -rf dist/
rm -rf build/
rm -rf pycollector.egg-info/
