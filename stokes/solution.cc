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
 * @ingroup Examples
 */

#include <deal.II/algorithms/newton.h>
#include <deal.II/base/flow_function.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <amandus/apps.h>
#include <amandus/stokes/function.h>
#include <amandus/stokes/matrix.h>
#include <amandus/stokes/solution.h>

#include <boost/scoped_ptr.hpp>

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);

  AmandusParameters param;
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  std::unique_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  //  Functions::StokesCosine<d> solution;
  //  Functions::PoisseuilleFlow<d> solution(1.,1.);
  StokesFlowFunction::StokesPolynomial<d> solution;

  StokesIntegrators::Matrix<d> matrix_integrator;
  StokesIntegrators::SolutionResidual<d> rhs_integrator(solution);
  StokesIntegrators::SolutionError<d> error_integrator(solution);

  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);
  ComponentMask boundary_components(d + 1, true);
  boundary_components.set(d, false);
  app.set_boundary(0, boundary_components);

  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);

  dealii::Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.parse_parameters(param);

  global_refinement_nonlinear_loop(6,
                                   app,
                                   newton,
                                   &error_integrator,
                                   static_cast<AmandusIntegrator<d>*>(nullptr),
                                   static_cast<dealii::Function<d>*>(nullptr),
                                   &solution,
                                   true);
}
