Set up your working directory (e.g. Laplace) outside the amandus where ever you want

 and put all the header files into include and the implentation files (.cc) files into source.

 Place all the files containing main method into main directory.

Then in working directory you make new directory e.g. build

mkdir build

cd build

cmake ..

For amandus, I use in a way that I separated the source and build in amandus by
making two directories inside amandus "source" and "build".

Now you just need to set the environment variables DEAL_II_DIR and AMANDUS_DIR
containing the path to dealii build or install location and amandus build
location, respectively. Or change paths directly in CMakeLists.txt file.

Rest of the guidelines you can find as comments in the CMakeLists.txt
