/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * <ul>
 * <li> Stationary Maxwell equations (divergence-free curl-curl problem)</li>
 * <li> Homogeneous tangential boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Linear solver</li>
 * </ul>
 *
 * @ingroup Maxwellgroup
 */

#include <amandus/apps.h>
#include <amandus/maxwell/matrix.h>
#include <amandus/rhs_one.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

int
main()
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);

  const unsigned int degree = 1;
  FE_Nedelec<d> vec(degree);
  FE_Q<d> scal(degree);
  FESystem<d> fe(vec, 1, scal, 1);

  MaxwellIntegrators::DivCurl::Matrix<d> matrix_integrator;
  RhsOne<d> rhs_integrator;

  AmandusApplication<d> app(tr, fe);
  app.set_boundary(0);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);

  global_refinement_linear_loop(5, app, solver, residual);
}
