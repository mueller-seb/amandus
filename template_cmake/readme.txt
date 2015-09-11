Set up your working directory (e.g. Laplace) out side the amandus where ever you want

 and put all the header files into include and .cc files into source.

Then in working directory you make new directory e.g. build

mkdir build

cd build

cmake ..


You need to make some changes of paths and target name in the CMakeLists.txt

file for this please read the comments in CMakeLists.txt file.
