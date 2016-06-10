/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * <ul>
 * <li> Laplace operator</li>
 * <li> Dirichlet boundary condition</li>
 * <li> Adaptive eigenvalue solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @author Joscha Gedicke
 *
 * @ingroup Examples
 */

#include <amandus/adaptivity.h>
#include <amandus/amandus_arpack.h>
#include <amandus/apps.h>
#include <amandus/laplace/eigen.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <boost/scoped_ptr.hpp>
#include <math.h>
#include <strings.h>

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);

  AmandusParameters param;
  param.declare_entry("MaxDofs", "1000", Patterns::Integer());
  param.declare_entry("Eigenvalue", "1", Patterns::Integer());
  param.declare_entry("ExactEigenvalue", "9.6397238440219", Patterns::Double());
  param.declare_entry("Domain", "L");
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_from_name<d>(param.get("FE")));
  param.leave_subsection();

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  if (strcasecmp(param.get("Domain").c_str(), "L") == 0)
    GridGenerator::hyper_L(tr, -1, 1);
  else
  {
    if (strcasecmp(param.get("Domain").c_str(), "slit") == 0)
      GridGenerator::hyper_cube_slit(tr, 0, 1);
    else
      GridGenerator::hyper_cube(tr, 0, 1);
  }
  param.enter_subsection("Discretization");
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  LaplaceIntegrators::Eigen<d> matrix_integrator;
  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);

  app.set_number_of_matrices(2);
  AmandusArpack<d> solver(app, matrix_integrator);

  LaplaceIntegrators::EigenEstimate<d> estimate_integrator;
  RefineStrategy::MarkBulk<d> refine_strategy(tr, 0.5);

  adaptive_refinement_eigenvalue_loop(param.get_integer("MaxDofs"),
                                      param.get_integer("Eigenvalue"),
                                      app,
                                      solver,
                                      estimate_integrator,
                                      refine_strategy,
                                      param.get_double("ExactEigenvalue"));
}