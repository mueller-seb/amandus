// $Id$

/**
 * @file Verify that DarcyMatrix and DarcyNoForceResidual are consistent
 */

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include "tests.h"
#include "darcy/noforce.h"
#include "darcy/matrix.h"

int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  
  const unsigned int degree = 1;
  FE_RaviartThomas<d> vec(degree);
  FE_DGQ<d> scal(degree);
  FESystem<d> fe(vec, 1, scal, 1);

  DarcyMatrix<d> matrix_integrator;
  DarcyNoForceResidual<d> rhs_integrator;
  
  AmandusApplication<d> app(tr, fe);
  
  verify_residual(5, app, matrix_integrator, rhs_integrator);
}
