// $Id$

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include "apps.h"
#include "heat/residual.h"
#include "heat/matrix.h"

// Exact polynomial solution to the Laplace problem
// Homogeneous no-slip boundary condition
// Linear solver

int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, 0, 1);
  
  const unsigned int degree = 2;
  FE_DGQ<d> fe(degree);

  /*  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(4, 1.);
  solution1d += Polynomials::Monomial<double>(2, -2.);
  solution1d += Polynomials::Monomial<double>(0, 1.);
  solution1d.print(std::cout); */
  
  HeatMatrix<d> matrix_integrator;
  HeatResidual<d> rhs_integrator;
  HeatError<d> error_integrator;
  
  AmandusApplication<d> app(tr, fe);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dof_handler);
  
  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.control.log_history(true);
  newton.control.set_reduction(1.e-14);
  newton.initialize(newout);
  newton.debug_vectors = true;
  newton.debug = 2;
  
  global_refinement_nonlinear_loop(10, app, newton, &error_integrator);
}
