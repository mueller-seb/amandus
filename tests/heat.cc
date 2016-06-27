/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 *
 * @brief Should be heat equation, but not working
 * <ul>
 * <li> Heat equations/li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Linear solver: due to the fact that we do not compute the
 * residual at the new time, but only invert the matrix, this method
 * is actually wrong except for theta close to zero</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */

#include <amandus/apps.h>
#include <amandus/laplace/matrix.h>
#include <amandus/laplace/polynomial.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

int
main()
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);

  Triangulation<d> tr;
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(3);

  const unsigned int degree = 2;
  FE_DGQ<d> fe(degree);

  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(4, 1.);
  solution1d += Polynomials::Monomial<double>(2, -2.);
  solution1d += Polynomials::Monomial<double>(0, 1.);
  solution1d.print(std::cout);

  LaplaceIntegrators::Matrix<d> matrix_integrator;
  LaplaceIntegrators::PolynomialResidual<d> rhs_integrator(solution1d);
  rhs_integrator.input_vector_names.push_back("Previous iterate");
  LaplaceIntegrators::PolynomialError<d> error_integrator(solution1d);

  AmandusApplication<d> app(tr, fe);
  app.set_boundary(0);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);

  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dofs());

  Algorithms::ThetaTimestepping<Vector<double>> timestepping(residual, solver);
  timestepping.set_output(newout);
  timestepping.theta(.75);

  timestepping.timestep_control().start_step(1.);
  timestepping.timestep_control().final(100.);
  global_refinement_nonlinear_loop(5, app, timestepping, &error_integrator);
}
