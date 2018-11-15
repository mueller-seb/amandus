/**
 * @defgroup Heatgroup Heat
 * @ingroup Examples
 *
 * This module describes the subdirectory `heat` of amandus.
 *
 * There are:
 * <ol>
 * <li>heat/matrix_heat.h containing the definition of the bilinear forms on cell and face level</li>
 * <li>heat/heat.cc: the driver program solving a simple heat problem</li>
 * <li>heat/heat_solution.cc: solve the heat equation with manufactured solution</li>
 * </ol>
 *
 *
 * <h3>heat/matrix_heat.h: the local contributions of the linear operator</h3>
 *
 * This file contains a single class LaplaceIntegrators::Matrix and the
 * definitions of its functions. Local integrators are derived from
 * AmandusIntegrator which in turn inherits
 * dealii::MeshWorker::LocalIntegrator. Its main purpose is the
 * implementation of the following three functions:
 *
 * The cell integrator function integrates the bilinear form of the
 * differential equation over a single cell. Using the predefined
 * integrators from deal.II, it is actually a one-liner here.
 *
 * \dontinclude laplace/matrix.h
 * \skip };
 * \skip template
 * \until }
 *
 * Since we are allowing for discontinuous Galerkin methods, we also
 * provide for integrators on boundary and interior faces. On the
 * boundary, we use Nitsche's method of weak boundary conditions,
 * already implemented in deal.II. If the finite element is
 * <i>H<sup>1</sup></i>-conforming, we skip the boundary integral. The
 * penalty parameter needed is computed automatically based on the
 * polynomial degree and cell geometry. Note that this is reliable
 * only on rectangular cells.
 * \until }
 *
 * On interior faces between two cells, we do the same with the
 * interior penalty method of Arnold. Note how the number of arguments
 * doubles, since we have two neighboring cells for such a face. Note
 * also, that there is no special code for hanging nodes. Those are
 * handled automatically by deal.II.
 * \until }
 *
 * <h3> laplace/laplace.cc: the basic example program</h3>
 *
 * The body of the main function has only 28 lines, and even these
 * have some redundancy for parameter files and logfile output. If you
 * did not change the parameter file (build/laplace/laplace.prm), it
 * computes a discontinuous Galerkin solution of a Poisson problem
 * with constant right hand side on a sequence of uniformly refined
 * meshes. The solutions are output into vtu files and can be
 * visualized in a viewer like paraview or visit. You can change the
 * output format in the parameter file; options can be found in the
 * deal.II DataOut documentation.
 *
 * Let us discuss the code in this file (skipping the documentation at
 * the beginning) line by line.  It begins at this point in the
 * file.
 *
 * \dontinclude laplace.cc
 * \skipline starts
 *
 * As usual, we have to include a bunch of header files in order
 * to get going.
 * \until {
 *
 * First, we define the space dimension, such that we can change the
 * computation do three dimensions easily.
 * \until const
 *
 * Then, we prepare the logfile to store our output. The logfile
 * contains all parameters given to the program as well as information
 * on the discretization, the solver, and output.
 * \until depth
 *
 * Next, we read the parameter file, which controls various aspects of
 * the run.
 * \until log
 *
 * The finite element to be used is taken from the parameter file. The
 * possible names are defined in the deal.II library. Note that this
 * program allows for continuous elements `FE_Q(k)` and for
 * discontinuous elements `FE_DGQ(k)`, where `k` is an integer number.
 * \until scoped
 *
 * We have to define a domain and its mesh, as well as initial
 * refinement, which is again taken from the parameter file.
 * \until leave
 *
 * Finally, we are ready to set up the actual simulation. Amandus uses
 * local integrators for that. The first integrator computes the
 * matrix cell by cell (and face by face for disconitunous Galerkin
 * methods). It is particular to the equation being solved and
 * therefore it is found in this namespace (Laplace). The right hand
 * side is a standard one (constant one), not related to the
 * Laplacian.
 * \until One
 *
 * The central object in every Amandus program is the application
 * class, typically of type AmandusApplication like here, using a
 * multigrid solver. In particular when you are developing code for a
 * new, complicated equation, you may want to replace this by the
 * direct solver application AmandusUmfpack. This application class
 * has to know about the parameters as well, and we set boundary
 * conditions.
 * \until boundary
 *
 * We use this application object to create a linear solver class and
 * a class for computing the right hand side of the linear system. The
 * latter is called "residual", since the residual in a Newton scheme
 * has the same structure. Check out AllenCahn for such an application.
 * \until residual
 *
 * Finally, we call one of the mesh refinement loops in apps.h. This
 * one solves linear problems on a sequence globally refined meshes.
 * \until }
 * 
 */
