/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * \brief Example for Laplacian with manufactured solution on adaptive meshes
 * <ul>
 * <li> Stationary Poisson equations</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Exact solution</li>
 * <li> Error computation</li>
 * <li> Error estimation</li>
 * <li> Adaptive linear solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @author Joscha Gedicke
 *
 * @ingroup Laplacegroup
 */

#include <amandus/adaptivity.h>
#include <amandus/apps.h>
#include <amandus/heat/matrix.h>
#include <amandus/heat/rhs.h>
#include <amandus/refine_strategy.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <boost/scoped_ptr.hpp>

//---------------------------------------------------------//

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  AmandusParameters param;
  param.declare_entry("MaxDofs", "1000", Patterns::Integer());
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  std::unique_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();


  HeatIntegrators::Matrix<d> matrix_integrator;
  HeatIntegrators::RHS<d> rhs_integrator;
  HeatIntegrators::Estimate<d> estimate_integrator;
  AmandusIntegrator<d>* error_integrator = 0;


  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  RefineStrategy::MarkBulk<d> refine_strategy(tr, 0.5);
  //RefineStrategy::MarkUniform<d> refine_strategy(tr);

  adaptive_refinement_linear_loop(param.get_integer("MaxDofs"),
                                  app,
                                  tr,
                                  solver,
                                  residual,
                                  estimate_integrator,
                                  refine_strategy, error_integrator);
}
