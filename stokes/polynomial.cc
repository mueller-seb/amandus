/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 *
 * @brief Stationary Stokes equations with a manufacured solution
 * <ul>
 * <li> Stationary Stokes equations</li>
 * <li> Homogeneous no-slip boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Linear solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */

#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <apps.h>
#include <stokes/polynomial.h>
#include <stokes/matrix.h>

#include <boost/scoped_ptr.hpp>

int main(int argc, const char** argv)
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  AmandusParameters param;
  param.read(argc, argv);
  param.log_parameters(deallog);
  
  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d> > fe(FETools::get_fe_from_name<d>(param.get("FE")));
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();
  
  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(4, 1.);
  solution1d += Polynomials::Monomial<double>(2, -2.);
  solution1d += Polynomials::Monomial<double>(0, 1.);
  solution1d.print(std::cout);
  
  Polynomials::Polynomial<double> solution1dp(1);
  solution1dp += Polynomials::Monomial<double>(3, 1.);
  
  StokesIntegrators::Matrix<d> matrix_integrator;
  StokesIntegrators::PolynomialRHS<d> rhs_integrator(solution1d, solution1dp);
  StokesIntegrators::PolynomialError<d> error_integrator(solution1d, solution1dp);
  
  AmandusApplicationSparseMultigrid<d> app(tr, *fe);
  ComponentMask boundary_components(d+1, true);
  boundary_components.set(d, false);
  app.set_boundary(0, boundary_components);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d>    residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);
  
  global_refinement_linear_loop(5, app, solver, residual, &error_integrator);
}