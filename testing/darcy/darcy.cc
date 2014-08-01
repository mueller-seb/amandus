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

int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  //tr.refine_global(1);
  
  const unsigned int degree = 1;
  FE_RaviartThomas<d> vec(degree);
  FE_DGQ<d> scal(degree);
  FESystem<d> fe(vec, 1, scal, 1);

  Polynomials::Polynomial<double> vector_potential;
  vector_potential += Polynomials::Monomial<double>(4, 1.);
  vector_potential += Polynomials::Monomial<double>(2, -2.);
  vector_potential += Polynomials::Monomial<double>(0, 1.);
  vector_potential.print(std::cout);

  Polynomials::Polynomial<double> scalar_potential;
  scalar_potential += Polynomials::Monomial<double>(3, -1.);
  scalar_potential += Polynomials::Monomial<double>(1, 3.);
  
  Polynomials::Polynomial<double> pressure_source(1);
//  pressure_source += Polynomials::Monomial<double>(3, 1.);
  
  DarcyMatrix<d> matrix_integrator;
  DarcyPolynomial::Residual<d> rhs_integrator(
      vector_potential, scalar_potential, pressure_source);
  DarcyPolynomial::Error<d> error_integrator(
      vector_potential, scalar_potential, pressure_source);
  
  AmandusApplicationSparseMultigrid<d> app(tr, fe);
  app.set_boundary(0);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);
  
  global_refinement_linear_loop(3, app, solver, residual);
}
