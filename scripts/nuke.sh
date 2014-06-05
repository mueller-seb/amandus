#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "run program as"
    echo "  nuke.sh /path/to/dealii [build-name]"
    echo "  where build-name defaults to 'build'"
    exit 1
fi

build="build"
if [ "$#" -gt 1 ]; then
    $build = $2;
fi

echo "Deleting and reconfiguring $build for deal.II path $1"
rm -rf $build
mkdir $build
cd $build
cmake -DDEAL_II_DIR=$1 ..
