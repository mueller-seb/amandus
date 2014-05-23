// $Id$

/**
 * @file
 * <ul>
 * <li> Heat equations/li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> </li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <apps.h>
#include <laplace/polynomial.h>
#include <laplace/matrix.h>

int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  
  const unsigned int degree = 4;
  FE_DGQ<d> fe(degree);

  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(4, 1.);
  solution1d += Polynomials::Monomial<double>(2, -2.);
  solution1d += Polynomials::Monomial<double>(0, 1.);
  solution1d.print(std::cout);
  
  LaplaceMatrix<d> matrix_integrator;
  LaplacePolynomialResidual<d> rhs_integrator(solution1d);
  rhs_integrator.input_vector_names.push_back("Previous iterate");
  LaplacePolynomialError<d> error_integrator(solution1d);
  
  AmandusApplication<d> app(tr, fe);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d>    residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);
  
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dof_handler);
  
  Algorithms::ThetaTimestepping<Vector<double> > timestepping(residual, solver);
  timestepping.set_output(newout);
  timestepping.theta(.55);
  
  timestepping.timestep_control().start_step(.1);
  timestepping.timestep_control().final(1.);
  global_refinement_nonlinear_loop(5, app, timestepping, &error_integrator);
}
