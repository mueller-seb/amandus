# README

Amandus is a simple experimentation suite built on the [dealii
library](http://www.dealii.org). It basically allows the
implementation of a new equation by just providing local integrators
for residuals and matrices. In addition, it has a lot of example
applications.

## Note: soon future change

Currently, Amandus uses the builtin dealii::Vector and
dealii::SparseMatrix objects. These will be replaced by templates
soon, such that parallelization and imported solvers will be possible.

## Legal note

Amandus is published under the [MIT License](./LICENSE.md).

When you commit a pull request to the amandus repository, you publish
your code together with the remaining parts of amandus under this
license. By submitting such a pull request, you certify that you or
the copyright owner agree to publish under this license.