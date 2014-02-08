// $Id$

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include "apps.h"
#include "stokes/polynomial.h"
#include "stokes/matrix.h"

// Exact polynomial solution to the Stokes problem
// Homogeneous no-slip boundary condition
// Linear solver

int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);

  const unsigned int degree = 3;
  FE_RaviartThomas<d> vec(degree);
  FE_DGQ<d> scal(degree);
  FESystem<d> fe(vec, 1, scal, 1);

  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(4, 1.);
  solution1d += Polynomials::Monomial<double>(2, -2.);
  solution1d += Polynomials::Monomial<double>(0, 1.);
  solution1d.print(std::cout);
  
  Polynomials::Polynomial<double> solution1dp(1);
  solution1dp += Polynomials::Monomial<double>(3, 1.);
  
  StokesMatrix<d> matrix_integrator;
  StokesPolynomialRHS<d> rhs_integrator(solution1d, solution1dp);
  StokesPolynomialError<d> error_integrator(solution1d, solution1dp);
  
  AmandusApplication<d> app(tr, fe, matrix_integrator, rhs_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);
  
  global_refinement_linear_loop(5, app, residual, &error_integrator);
}
