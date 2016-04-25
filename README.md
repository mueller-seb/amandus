# README

Amandus is a simple experimentation suite built on the [dealii
library](http://www.dealii.org). It basically allows the
implementation of a new equation by just providing local integrators
for residuals and matrices. In addition, it has a lot of example
applications. You can find more information in [the documentation generated
by doxygen](http://pesser.bitbucket.org/amandus_doc/).

## Getting started

### Prerequisites

Amandus is updated along with the [development version of
deal.II](https://github.com/dealii/dealii). Thus, first make sure to obtain
a recent clone of it and build it with Arpack and UMFPACK enabled. If your
system has those libraries installed, it should be as simple as
```
git clone https://github.com/dealii/dealii.git dealii_src
mkdir dealii_build
cd dealii_build
cmake -DDEAL_II_WITH_ARPACK=ON -DDEAL_II_WITH_UMFPACK=ON ../dealii_src
make -j <number_of_cores>
```
Notice that UMFPACK is not strictly required to build amandus, but
nevertheless it is very useful and used by many examples of Amandus. The
deal.II website provides [more information about installing
Arpack](https://dealii.org/developer/external-libs/arpack.html).

### Preparing Amandus

Amandus uses [cmake](http://www.cmake.org/) for configuration. It supports
in and out of source builds. In-source builds are not recommended if you
plan on updating or developing. Thus, a typical setup runs like this
```
git clone git@bitbucket.org:guidokanschat/amandus.git amandus_src
mkdir amandus_build
cd amandus_build
cmake -DDEAL_II_DIR=</path/to/dealii> ../amandus_src
```
where `<path/to/dealii>` is the `dealii_build` directory from the previous
step. There are two common workflows for using Amandus. The first one starts
from one of the examples included with Amandus and gets you started quickly,
the other one uses Amandus as a library for more complex projects where it
might be preferable to keep the project's code seperated from Amandus' code.

### Included Examples

After the steps of the previous section, your `amandus_build` directory will
contain different folders with names of specific problems (e.g. `laplace`,
`maxwell`, `allen_cahn`, etc...). If you are interested in one of those
examples you can `cd` into its folder and call `make` to build it.
Typically, this will create multiple executables as well as parameter files
(ending with `.prm`) that are executed as `./example_name example_name.prm`.
If one of the examples seems to be a good starting point for your own model
you can just go ahead and adapt the code in the `amandus_src` directory to
your needs and see the results of your changes by another call to `make` in
the example's build directory (i.e. `amandus_build/example`).


### Using Amandus as a library

If you want to use Amandus as an external dependency for your own code, you
would call `make` in the `amandus_build` directory to build the library.
Optionally you can install it from the build directory with
```
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/amandus/to .
make install
```

In order to use Amandus for your own project you can use a setup similiar to
the example in `template_cmake` directory which would be compiled as
```
cmake -DAMANDUS_DIR=/path/to/amandus/build/or/install /path/to/your/project/source
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
