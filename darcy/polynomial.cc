/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 *
 * @brief Stationary Darcy equations with a manufacured solution
 * <ul>
 * <li> Stationary Darcy equations</li>
 * <li> Homogeneous no-slip boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Linear solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Darcygroup
 */

#include <amandus/apps.h>
#include <amandus/darcy/matrix.h>
#include <amandus/darcy/polynomial.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <boost/scoped_ptr.hpp>

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  AmandusParameters param;
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  if (param.get_bool("Local refinement"))
    {
      tr.refine_global(1);
      tr.begin_active()->set_refine_flag();
      tr.execute_coarsening_and_refinement();
    }
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();
  
  // The curl potentialof u, needs zero tangential derivatives at the
  // boundary for consistency with boundary conditions
  Polynomials::Polynomial<double> solution1dcurl(4);
  solution1dcurl += Polynomials::Monomial<double>(4, 1.);
  solution1dcurl += Polynomials::Monomial<double>(2, -2.);
  solution1dcurl += Polynomials::Monomial<double>(0, 1.);
  solution1dcurl.print(std::cout);

  // The following part is not working yet. Solutions seem to havr the
  // same shape, there holds u=grad p, but there is a factor between
  // the true and numerical solution.
  
  // The grad potential of u, needs zero normal derivatives at the boundary
  Polynomials::Polynomial<double> solution1dgrad(3);
  // solution1dgrad += Polynomials::Monomial<double>(3, 1.);
  // solution1dgrad += Polynomials::Monomial<double>(1, 3.);

  // The gradient potential of f, source for nonzero pressure
  Polynomials::Polynomial<double> solution1dp(3);
  solution1dp += Polynomials::Monomial<double>(3, 1.);

  DarcyIntegrators::Matrix<d> matrix_integrator;
  DarcyIntegrators::Polynomial::RHS<d> rhs_integrator(solution1dcurl, solution1dgrad, solution1dp);
  DarcyIntegrators::Polynomial::Error<d> error_integrator(solution1dcurl, solution1dgrad, solution1dp);;

  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);
  ComponentMask boundary_components(d + 1, true);
  boundary_components.set(d, false);
  app.set_boundary(0, boundary_components);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);

  global_refinement_linear_loop(
    param.get_integer("Steps"), app, solver, residual, &error_integrator);
}
