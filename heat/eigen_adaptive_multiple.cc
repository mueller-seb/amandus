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
 * <li> Adaptive eigenvalue solver for double eigenvalue</li>
 * <li> UMFPACK</li>
 * </ul>
 *
 * @author Joscha Gedicke
 *
 * @ingroup Laplacegroup
 */

#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <amandus/adaptivity.h>
#include <amandus/amandus_arpack.h>
#include <amandus/apps.h>
#include <amandus/laplace/eigen.h>

#include <boost/scoped_ptr.hpp>

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  AmandusParameters param;
  param.declare_entry("MaxDofs", "1000", Patterns::Integer());
  param.declare_entry("Eigenvalue", "2", Patterns::Integer());
  param.declare_entry("Multiplicity", "2", Patterns::Integer());
  param.declare_entry("ExactEigenvalue", "9.39083794", Patterns::Double());
  param.enter_subsection("Arpack");
  param.set("Symmetric", "true");
  param.leave_subsection();
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  std::unique_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));
  param.leave_subsection();

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::cheese(tr, std::vector<unsigned int>(d, 1));
  param.enter_subsection("Discretization");
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  LaplaceIntegrators::Eigen<d> matrix_integrator;
  AmandusUMFPACK<d> app(tr, *fe);
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
                                      param.get_double("ExactEigenvalue"),
                                      param.get_integer("Multiplicity"),
                                      3);
}
