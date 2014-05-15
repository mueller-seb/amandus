// $Id$

#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <apps.h>
#include <darcy/polynomial.h>
#include <darcy/matrix.h>

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
  
  const unsigned int degree = 1;
  FE_BDM<d> vec(degree+1);
  FE_DGP<d> scal(degree);
  FESystem<d> fe(vec, 1, scal, 1);

  Polynomials::Polynomial<double> vector_potential;
  vector_potential += Polynomials::Monomial<double>(4, 1.);
  vector_potential += Polynomials::Monomial<double>(2, -2.);
  vector_potential += Polynomials::Monomial<double>(0, 1.);
  vector_potential.print(std::cout);

  Polynomials::Polynomial<double> scalar_potential(1);
  // scalar_potential += Polynomials::Monomial<double>(3, -1.);
  // scalar_potential += Polynomials::Monomial<double>(1, 3.);
  
  Polynomials::Polynomial<double> pressure_source(1);
//  pressure_source += Polynomials::Monomial<double>(3, 1.);
  
  DarcyMatrix<d> matrix_integrator;
  DarcyPolynomial::Residual<d> rhs_integrator(vector_potential, scalar_potential, pressure_source);
  DarcyPolynomial::Error<d> error_integrator(vector_potential, scalar_potential, pressure_source);
  
  AmandusApplication<d> app(tr, fe);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d>    residual(app, rhs_integrator);
  
  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.control.log_history(true);
  newton.control.set_reduction(1.e-14);
  newton.control.set_tolerance(1.e-5);
  newton.threshold(.1);
  
  global_refinement_nonlinear_loop(5, app, newton, &error_integrator);
}
