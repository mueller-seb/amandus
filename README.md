# README

Amandus is a simple experimentation suite built on the [dealii
library](http://www.dealii.org). It basically allows the
implementation of a new equation by just providing local integrators
for residuals and matrices. In addition, it has a lot of example
applications. You can find more information in [the documentation generated
by doxygen](http://pesser.bitbucket.org/amandus_doc/).

## Building and installing

Amandus uses [cmake](http://www.cmake.org/) for configuration. It supports in and out of source builds, but not installing. In-source buids are not recommended if you plan on updating or developing. Thus, a typical setup runs like this 
```
cd build
cmake -DDEAL_II_DIR=/path/to/installed/dealii /path/to/amandus/source
make
```

## Buildung a project for Eclipse with cmake
After building and installing go your project folder in the build directory (e.g. /path/to/build/allen_cahn/) and use:

cmake -"GEclipse CDT4 - Unix Makefiles" -DDEAL_II_DIR=/path/to/installed/dealii /path/to/amandus/source/project/. 

Note the "." at the end. Note the spelling Eclipse flag. 
That command creates a .project file in the build folder that you can import into Eclipse via: 
"File">"Import">"Existing Projects Into Workspace">Select the project folder in the build directory > "Finish"
Then you might have to add the path to the deal.II and amandus source, such that Eclipse finds everything:
RKlick the project on the left side>"Properties">"C/C++ Include Path ...">"Add Folder/File"
1) /.../deal.ii/include
2) /.../amandus
After adding these it might be necessary to rerun the Indexer:
RKlick the project on the left side>"Index">"Rebuild

If you want to compile and run in Eclipse then you can set up everything as usual. See deal.II wiki, section Eclipse.



## Note: soon future change

Currently, Amandus uses the builtin `dealii::Vector` and
`dealii::SparseMatrix` objects. These will be replaced by templates
soon, such that parallelization and imported solvers will be possible.

## Legal note

Amandus is published under the [MIT License](./LICENSE.md).

When you commit a pull request to the amandus repository, you publish
your code together with the remaining parts of amandus under this
license. By submitting such a pull request, you certify that you or
the copyright owner agree to publish under this license.
