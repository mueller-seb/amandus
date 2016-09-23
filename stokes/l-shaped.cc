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
 * <li> Homogeneous slip boundary condition</li>
 * <li> Exact FlowFunction solution</li>
 * <li> Nonlinear solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @author Arbaz Khan
 *
 * @ingroup Examples
 */
#include <amandus/adaptivity.h>
#include <amandus/apps.h>
#include <amandus/stokes/function.h>
#include <amandus/stokes/matrix.h>
#include <amandus/stokes/solution.h>
#include <amandus/refine_strategy.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/base/flow_function.h>
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

  AmandusParameters param;
  param.declare_entry("MaxDofs", "1000", Patterns::Integer());
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_L(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  Functions::StokesLSingularity solution;

  StokesIntegrators::Matrix<d> matrix_integrator;
  StokesIntegrators::SolutionResidual<d> rhs_integrator(solution);
  rhs_integrator.input_vector_names.push_back("Newton iterate");
  StokesIntegrators::SolutionError<d> error_integrator(solution);
  StokesIntegrators::SolutionEstimate<d> SolutionEstimate(solution);

  // AmandusApplication<d> app(tr, *fe);
  AmandusUMFPACK<d> app(tr, *fe);
  app.parse_parameters(param);
  ComponentMask boundary_components(d + 1, true);
  ComponentMask pressure_component(d + 1, false);
  boundary_components.set(d, false);
  pressure_component.set(d, true);
  app.set_boundary(0, boundary_components);
  app.set_meanvalue(pressure_component);
  // app.parse_parameters(param);

  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  // RefineStrategy::MarkBulk<d> refine_strategy(tr, 0.5);
  RefineStrategy::MarkUniform<d> refine_strategy(tr);

  dealii::Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.parse_parameters(param);

  adaptive_refinement_nonlinear_loop(param.get_integer("MaxDofs"),
                                     app,
                                     tr,
                                     newton,
                                     SolutionEstimate,
                                     refine_strategy,
                                     &error_integrator,
                                     &solution,
                                     true);
}
