/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * <ul>
 * <li> Elasticity operator</li>
 * <li> Neumann boundary condition</li>
 * <li> Eigenvalue problem</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Elasticitygroup
 */

#include <amandus/amandus_arpack.h>
#include <amandus/apps.h>
#include <amandus/elasticity/eigen.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <elasticity/parameters.h>

#include <boost/scoped_ptr.hpp>

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  AmandusParameters param;
  ::Elasticity::Parameters::declare_parameters(param);
  param.declare_entry("Eigenvalues", "12", Patterns::Integer());
  param.enter_subsection("Arpack");
  param.set("Symmetric", "true");
  param.leave_subsection();
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();
  std::set<unsigned int> boundaries;
  boundaries.insert(0);

  ::Elasticity::Parameters parameters;
  parameters.parse_parameters(param);
  Elasticity::Eigen<d> matrix_integrator(parameters, boundaries);
  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);
  app.boundary_patches = true;

  app.set_number_of_matrices(2);
  AmandusArpack<d> solver(app, matrix_integrator);

  global_refinement_eigenvalue_loop(
    param.get_integer("Steps"), param.get_integer("Eigenvalues"), app, solver);
}
