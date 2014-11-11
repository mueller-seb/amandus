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
#include <advection-diffusion/matrix_deal.h>

//test
#include <deal.II/base/parameter_handler.h>

#include <boost/scoped_ptr.hpp>


int main(int argc, const char** argv)
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  AmandusParameters param;
  ::AdvectionDiffusion::Parameters::declare_parameters(param);
  //param.read(argc, argv);
  //macht das selbe wir param.read, nur ohne amandus-funktion, sondern direkt mit deal-parameter_handler
  // param.read sucht das .prm-file, hier bei read_input ist das .prm mit namen angegeben.
  param.read_input("polynomial_01.prm", true);
  param.log_parameters(deallog);
  
  param.enter_subsection("Discretization");
  //FE         = FE_DGQ(2);
  //Refinement = 0;
  boost::scoped_ptr<const FiniteElement<d> > fe(FETools::get_fe_from_name<d>(param.get("FE")));
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();
  
  // Polynomials for the right-hand-side (not used in every setting, see polynomial_boundary.h)
  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(2, -1.);
  solution1d += Polynomials::Monomial<double>(0, 1.);  
  //solution1d += Polynomials::Monomial<double>(1, -1.);
  //solution1d += Polynomials::Monomial<double>(0, 1.);  
  //solution1d += Polynomials::Monomial<double>(0, 0);
  //solution1d += Polynomials::Monomial<double>(0, 0);  
  solution1d.print(std::cout);
  
  std::vector<Polynomials::Polynomial<double> > potentials(1);
  potentials[0] = solution1d;

  // factors belonging to the diffusion term
  //double factor1= 0.001;
  //double factor2=3;
  double factor1= 0.001;
  double factor2= 1;
  
  // obstacle (where factor2 holds)
  double x1 = -0.5;
  double x2 = 0.0;
  double y1 = -0.5;
  double y2 = 0.0;
  
  // Direction of the velocity (advection term)
  std::vector<std::vector<double> > direction(d,std::vector<double>(1));
  direction[0][0] = 0.1;
  direction[1][0] = 0.2;
 
 
  ::AdvectionDiffusion::Parameters parameters;
  parameters.parse_parameters(param);
  ::AdvectionDiffusion::Matrix<d> matrix_integrator(parameters, factor1, factor2, direction, x1, x2, y1, y2);
  ::AdvectionDiffusion::PolynomialBoundaryRHS<d> rhs_integrator(parameters, potentials, factor1, factor2, direction, x1, x2, y1, y2);
  //::AdvectionDiffusion::PolynomialBoundaryError<d> error_integrator(parameters, potentials);
  
  
  AmandusUMFPACK<d>  app(tr, *fe);
  AmandusSolve<d>    solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);
  
  //global_refinement_linear_loop(5, app, solver, residual, &error_integrator);
  global_refinement_linear_loop(5, app, solver, residual);
  
}
