// $Id$

/**
 * @file
 * <ul>
 * <li> Stationary Maxwell equations (curl-curl problem)</li>
 * <li> Homogeneous tangential boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Linear solver</li>
 * </ul>
 *
 * @ingroup Examples
 */

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <apps.h>
#include <maxwell/matrix.h>
#include <rhs_one.h>


int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);

  const unsigned int degree = 1;
  FE_Nedelec<d> vec(degree);
  FE_Q<d> scal(degree);
  FESystem<d> fe(vec, 1, scal, 1);
  
  CurlCurlMatrix<d> matrix_integrator;
  RhsOne<d> rhs_integrator;
  
  AmandusApplicationSparseMultigrid<d> app(tr, fe);
  app.set_boundary(0);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d>    residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);
  
  global_refinement_linear_loop(5, app, solver, residual);
}
