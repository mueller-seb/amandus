// $Id$

#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include "apps.h"
#include "darcy/polynomial.h"
#include "darcy/matrix.h"

// Exact polynomial solution to the Darcy problem
// Homogeneous no-slip boundary condition
// Linear solver

int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  tr.refine_global(3);
  
  const unsigned int degree = 3;
  FE_BDM<d> vec(degree);
  FE_DGP<d> scal(degree);
  FESystem<d> fe(vec, 1, scal, 1);

  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(4, 1.);
  solution1d += Polynomials::Monomial<double>(2, -2.);
  solution1d += Polynomials::Monomial<double>(0, 1.);
  solution1d.print(std::cout);
  
  Polynomials::Polynomial<double> solution1dp(1);
//  solution1dp += Polynomials::Monomial<double>(3, 1.);
  
  DarcyMatrix<d> matrix_integrator;
  DarcyPolynomialResidual<d> rhs_integrator(solution1d, solution1d);
  DarcyPolynomialError<d> error_integrator(solution1d, solution1d);
  
  AmandusApplication<d> app(tr, fe);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d>    residual(app, rhs_integrator);
  
  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.control.log_history(true);
  newton.control.set_reduction(1.e-14);
  newton.threshold(.1);
  
  global_refinement_nonlinear_loop(5, app, newton, &error_integrator);
}
