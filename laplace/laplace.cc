/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * \file
 * \brief Example program with constant right hand side
 * \ingroup Laplacegroup
 * <ul>
 * <li> Stationary Poisson equation</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Constant right hand side</li>
 * <li> Linear solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * This is a simple usage example of the classes in this
 * directory.
 */

// ********************************************************************** //
// Program starts here

#include <amandus/apps.h>
#include <amandus/laplace/matrix.h>
#include <amandus/rhs_one.h>
#include <deal.II/fe/fe_tools.h>

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
  GridGenerator::hyper_cube_slit(tr, -1., 1.);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  LaplaceIntegrators::Matrix<d> matrix_integrator;
  RhsOne<d> rhs_integrator(1);

  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);

  global_refinement_linear_loop(param.get_integer("Steps"), app, solver, residual);
}
