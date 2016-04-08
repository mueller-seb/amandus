/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * Verify that StokesIntegrators::Matrix and StokesIntegrators::NoForceResidual are consistent
 *
 * @ingroup Verification
 */

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <amandus/tests.h>
#include <amandus/stokes/noforce.h>
#include <amandus/stokes/matrix.h>

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

  StokesIntegrators::Matrix<d> matrix_integrator;
  StokesIntegrators::NoForceResidual<d> rhs_integrator;
  
  AmandusApplicationSparse<d> app(tr, fe);
  
  verify_residual(5, app, matrix_integrator, rhs_integrator);
}
