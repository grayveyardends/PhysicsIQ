#!/bin/sh
# builds libgeomkit.so next to this script. no deps beyond g++.
set -e
cd "$(dirname "$0")"
g++ -O2 -march=native -shared -fPIC -o libgeomkit.so geomkit.cpp
echo "built $(pwd)/libgeomkit.so"
