/**********************************************************************
 *  Copyright (C) 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 *
 * <h3> About this example</h3>
 *
 * This program build on <tt>laplace.cc</tt> in this same directory,
 * but it adds error estimation and adaptively refined meshes.
 *
 * @ingroup Laplacegroup
 *
 *  <h3> How to compile and run </h3>
 *
 *  Go to <code>example/laplace</code> directory and type:
 * @code
 * make laplace_laplace
  @endcode
 * Now you can see the executable. To execute it and produce the output, type:
 * @code
 * ./laplace
 * @endcode
 *
 *
 * <h3> Introduction </h3>
 *
 * We solve the Laplace equation
 *
 * \f{align*}
 * - \Delta u & = 0 \qquad\qquad & \text{in}\ \Omega
 * \f}
 *
 * on the L-shaped domain \f$\Omega=[-1,1]^2 \setminus [0,1]^2\f$.
 *
 * @ingroup Laplacegroup
 */

#include <amandus/apps.h>
#include <amandus/laplace/matrix.h>
#include <amandus/laplace/matrix_factor.h>
#include <amandus/laplace/noforce.h>
#include <amandus/laplace/solution.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/algorithms/newton.h>

template <int dim>
class Startup : public dealii::Function<dim>
{
public:
  Startup();
  virtual double value(const Point<dim>& p, const unsigned int component) const;

  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component) const;
};

/**
 * We fix the dimension of the domain
 * @code
 * const unsigned int d=2;
 * @endcode
 * Construction of a <code>ofstream</code> and the attachment of
 * the stream to the open file <code>deallog</code>
 * @code
 * std::ofstream logfile("deallog");
 * deallog.attach(logfile);
 * @endcode
 * Now we define an object for a triangulation and we fill
 * it with a single cell of a square domain. The triangulation is
 * refined 3 times, to yield \f$4^3=64\f$ cells in total.
 * @code
 * Triangulation<d> tr;
 * GridGenerator::hyper_cube (tr, -1, 1);
 * tr.refine_global(3);
 * @endcode
 * We construct the tensor product polynomials of degree p. We use
 * continuous polynomials.
 * @code
 * const unsigned int degree = 2;
 * FE_Q<d> fe(degree);
 * @endcode
 * Creation of a Matrix object. The Matrix class permits to construct the
 * matrix A using LocalIntegrators functions. It is essential to implement
 * a class matrix for every kind of problem you want to solve. For more
 * details, read the documentation for the matrix class of the Laplace
 * problem.
 * @code
 * LaplaceIntegrators::Matrix<d> matrix_integrator;
 * @endcode
 *
 * @todo The code does't compile. Correct it and finish the code description.
 */
int
main()
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_L(tr);
  tr.refine_global(1);

  const unsigned int degree = 2;
  FE_Q<d> fe(degree);

  LaplaceIntegrators::Matrix<d> matrix_integrator;
  matrix_integrator.use_boundary = false;
  LaplaceIntegrators::NoForceResidual<d> rhs_integrator;
  rhs_integrator.use_boundary = false;

  AmandusUMFPACK<d> app(tr, fe);
  app.set_boundary(0);

  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);

  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dofs());

  Algorithms::Newton<Vector<double>> newton(residual, solver);
  // newton.parse_parameters(param);

  newton.initialize(newout);
  newton.debug_vectors = true;

  Functions::LSingularityFunction exact;
  LaplaceIntegrators::SolutionEstimate<d> estimator(exact);
  LaplaceIntegrators::SolutionError<d> error_integrator(exact);

  RefineStrategy::MarkBulk<d> refine_strategy(tr, 0.5);

  if (false)
    global_refinement_nonlinear_loop(5, app, newton, &error_integrator, &estimator, &exact);
  else
    adaptive_refinement_nonlinear_loop(
      1000000, app, tr, newton, estimator, refine_strategy, &error_integrator, &exact);
}
