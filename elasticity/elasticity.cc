// $Id$

/**
 * @file
 * <ul>
 * <li> Stationary Elasticity equations</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Newton solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */

#include <deal.II/fe/fe_tools.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <apps.h>
#include <elasticity/parameters.h>
#include <elasticity/residual.h>
#include <elasticity/matrix.h>

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
  
  Elasticity::Matrix<d> matrix_integrator;
  Elasticity::Residual<d> rhs_integrator(solution1d);
  rhs_integrator.input_vector_names.push_back("Newton iterate");
  
  AmandusApplication<d> app(tr, *fe);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d>    residual(app, rhs_integrator);
  
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dof_handler);
  
  param.enter_subsection("Newton");
  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.parse_parameters(param);
  newton.initialize(newout);
  param.leave_subsection();
  
  global_refinement_nonlinear_loop(5, app, newton, &error_integrator);
}