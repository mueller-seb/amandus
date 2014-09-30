/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * <ul>
 * <li> Stationary advection-diffusion equations</li>
 * <li> Inhomogeneous Dirichlet boundary condition</li>
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
#include <advection-diffusion/polynomial_boundary.h>
#include <advection-diffusion/matrix.h>


#include <boost/scoped_ptr.hpp>


int main(int argc, const char** argv)
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  AmandusParameters param;
  ::Advection::Parameters::declare_parameters(param);
  param.read(argc, argv);
  param.log_parameters(deallog);
  
  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d> > fe(FETools::get_fe_from_name<d>(param.get("FE")));
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();
  
  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(2, -1.);
  solution1d += Polynomials::Monomial<double>(0, 1.);            
  solution1d.print(std::cout);
  
  std::vector<Polynomials::Polynomial<double> > potentials(1);
  potentials[0] = solution1d;

  double faktor=2;
  std::vector<std::vector<double> > direction(d,std::vector<double>(1));
  direction[0][0] = 1.;
  direction[1][0] = 2.;
 
  ::Advection::Parameters parameters;
  parameters.parse_parameters(param);
  ::Advection::Matrix<d> matrix_integrator(parameters, faktor, direction);
  ::Advection::PolynomialBoundaryRHS<d> rhs_integrator(parameters, potentials, faktor, direction);
  ::Advection::PolynomialBoundaryError<d> error_integrator(parameters, potentials);
  
  
  AmandusUMFPACK<d>  app(tr, *fe);
  AmandusSolve<d>    solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);
  
  global_refinement_linear_loop(5, app, solver, residual, &error_integrator);
}
