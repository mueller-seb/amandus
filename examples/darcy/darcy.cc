/**
 * @file
 *
 * Exact polynomial solution to the Darcy problem
 * Homogeneous no-slip boundary condition
 * Linear solver
 *
 * @ingroup Examples
 */

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <apps.h>
#include <darcy/polynomial.h>
#include <darcy/matrix.h>
#include <deal.II/fe/fe_tools.h>

int main(int argc, const char** argv)
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);

  AmandusParameters param;
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");

  const FiniteElement<d>* fe(
      FETools::get_fe_from_name<d>(param.get("FE")));
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();
  
  Polynomials::Polynomial<double> vector_potential;
  vector_potential += Polynomials::Monomial<double>(4, 1.);
  vector_potential += Polynomials::Monomial<double>(2, -2.);
  vector_potential += Polynomials::Monomial<double>(0, 1.);
  vector_potential.print(std::cout);

  Polynomials::Polynomial<double> scalar_potential;
  scalar_potential += Polynomials::Monomial<double>(3, -1.);
  scalar_potential += Polynomials::Monomial<double>(1, 3.);
  
  Polynomials::Polynomial<double> pressure_source(1);
  pressure_source += Polynomials::Monomial<double>(3, 1.);
  
  DarcyMatrix<d> matrix_integrator;
  DarcyPolynomial::RHS<d> rhs_integrator(
      vector_potential, scalar_potential, pressure_source);
  DarcyPolynomial::Error<d> error_integrator(
      vector_potential, scalar_potential, pressure_source);
  
  AmandusApplicationSparseMultigrid<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);
  
  global_refinement_linear_loop(2, app, solver, residual, &error_integrator);
}
