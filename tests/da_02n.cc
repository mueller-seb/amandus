/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 *
 * @brief Polynomial solution of Darcy equations
 * <ul>
 * <li>Exact polynomial solution to the Darcy problem</li>
 * <li>BDM elements</li>
 * <li>Homogeneous no-slip boundary condition</li>
 * <li>Linear solver</li>
 * </ul>
 *
 * @ingroup Examples
 */
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <amandus/apps.h>
#include <amandus/darcy/polynomial/polynomial.h>
#include <amandus/darcy/integrators.h>


int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  tr.refine_global(1);
  
  const unsigned int degree = 1;
  FE_BDM<d> vec(degree+1);
  FE_DGP<d> scal(degree);
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
  
  Darcy::SystemIntegrator<d> matrix_integrator;
  Darcy::Polynomial::Residual<d> rhs_integrator(vector_potential, scalar_potential, pressure_source);
  rhs_integrator.input_vector_names.push_back("Newton iterate");
  Darcy::Polynomial::Error<d> error_integrator(vector_potential, scalar_potential, pressure_source);
  
  AmandusApplicationSparseMultigrid<d> app(tr, fe);
  app.set_boundary(0);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d>    residual(app, rhs_integrator);
  
  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.control.log_history(true);
  newton.control.set_reduction(1.e-14);
  newton.threshold(.1);
  
  global_refinement_nonlinear_loop(5, app, newton, &error_integrator);
}
