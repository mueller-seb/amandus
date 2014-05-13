// $Id$

/**
 * @file
 * Verify that LaplaceMatrix and LaplaceNoForceResidual are consistent
 */

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include "tests.h"
#include "laplace/noforce.h"
#include "laplace/matrix.h"

int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  
  const unsigned int degree = 1;
  FE_DGQ<d> fe(degree);

  LaplaceMatrix<d> matrix_integrator;
  LaplaceNoForceResidual<d> rhs_integrator;
  
  AmandusApplication<d> app(tr, fe);
  
  verify_residual(5, app, matrix_integrator, rhs_integrator);
}
