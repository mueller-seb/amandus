/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * Verify that LaplaceIntegrators::Matrix and LaplaceIntegrators::NoForceResidual are consistent
 *
 * @ingroup Verification
 */

#include <amandus/laplace/matrix.h>
#include <amandus/laplace/noforce.h>
#include <amandus/tests.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

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
  param.leave_subsection();

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);

  LaplaceIntegrators::Matrix<d> matrix_integrator;
  LaplaceIntegrators::NoForceResidual<d> rhs_integrator;

  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);

  verify_residual(5, app, matrix_integrator, rhs_integrator);
}
