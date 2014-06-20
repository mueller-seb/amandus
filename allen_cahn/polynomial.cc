// $Id$

/**
 * @file
 *
 * @brief Stationary Allen-Cahn with manufactured solution
 * <ul>
 * <li>Stationary Allen-Cahn equations</li>
 * <li>Homogeneous Dirichlet boundary conditions</li>
 * <li>Exact polynomial solutionExact polynomial solution</li>
 * <li>Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <apps.h>
#include <allen_cahn/polynomial.h>
#include <allen_cahn/matrix.h>


int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  tr.refine_global(3);
  
  const unsigned int degree = 3;
  FE_DGQ<d> fe(degree);

  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(3, -1.);
  solution1d += Polynomials::Monomial<double>(1, 3.);
  solution1d.print(std::cout);
  
  AllenCahn::Matrix<d> matrix_integrator(1.);
  matrix_integrator.input_vector_names.push_back("Newton iterate");
  AllenCahn::PolynomialResidual<d> rhs_integrator(1., solution1d);
  AllenCahn::PolynomialError<d> error_integrator(solution1d);
  
  AmandusApplicationSparseMultigrid<d> app(tr, fe);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dofs());
  
  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.control.log_history(true);
  newton.control.set_reduction(1.e-14);
  newton.threshold(.2);
  
  // newton.initialize(newout);
  // newton.debug_vectors = true;
  // newton.debug = 2;
  
  global_refinement_nonlinear_loop(5, app, newton, &error_integrator);
}
