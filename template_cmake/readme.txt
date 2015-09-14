Set up your working directory (e.g. Laplace) outside the amandus where ever you want

 and put all the header files into include and .cc files into source.

Then in working directory you make new directory e.g. build

mkdir build

cd build

cmake ..

For amandus, I use in a way that I separated the source and build in amandus by
making two directories inside amandus "source" and "build".

Then set the envoirnment variable AMANDUS by

export AMANDUS=/path/to/amandus

Now you just need to change path for dealii install directory in CMakeLists.txt file.
