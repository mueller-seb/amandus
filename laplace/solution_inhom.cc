/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * <ul>
 * <li> Stationary Poisson equations</li>
 * <li> Inhomogeneous Dirichlet boundary condition</li>
 * <li> Exact (singular) solution</li>
 * <li> Adaptive linear solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @author Joscha Gedicke
 *
 * @ingroup Examples
 */

#include <amandus/adaptivity.h>
#include <amandus/apps.h>
#include <amandus/laplace/matrix.h>
#include <amandus/laplace/solution.h>
#include <amandus/refine_strategy.h>
#include <deal.II/algorithms/newton.h>
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
  param.read_input("solution_inhom.prm", true);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_from_name<d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_L(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  Functions::LSingularityFunction exact_solution;

  LaplaceIntegrators::Matrix<d> matrix_integrator;
  LaplaceIntegrators::SolutionResidual<d> rhs_integrator(exact_solution);
  rhs_integrator.input_vector_names.push_back("Newton iterate");
  LaplaceIntegrators::SolutionError<d> error_integrator(exact_solution);

  LaplaceIntegrators::SolutionEstimate<d> estimate_integrator(exact_solution);

  AmandusApplication<d> app(tr, *fe);
  app.set_boundary(0);
  app.parse_parameters(param);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  RefineStrategy::MarkBulk<d> refine_strategy(tr, 0.5);

  Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.parse_parameters(param);

  adaptive_refinement_nonlinear_loop(param.get_integer("MaxDofs"),
                                     app,
                                     tr,
                                     newton,
                                     estimate_integrator,
                                     refine_strategy,
                                     &error_integrator,
                                     &exact_solution);
}