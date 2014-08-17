/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * Verify that LaplaceMatrix and LaplaceNoForceResidual are consistent
 *
 * @ingroup Verification
 */

#include <deal.II/fe/fe_tools.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <tests.h>
#include <laplace/noforce.h>
#include <laplace/matrix.h>

using namespace Integrators;

int main(int argc, const char** argv)
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  AmandusParameters param;
  param.read(argc, argv);
  param.log_parameters(deallog);
  
  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d> > fe(FETools::get_fe_from_name<d>(param.get("FE")));
  param.leave_subsection();
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  
  LaplaceMatrix<d> matrix_integrator;
  Theta<d> mi(matrix_integrator, true);
  LaplaceNoForceResidual<d> rhs_integrator;
  Theta<d> ri(rhs_integrator, true);
  ri.input_vector_names.push_back("Newton iterate");
  
  AmandusApplicationSparseMultigrid<d> app(tr, *fe);
  
  verify_theta_residual(5, app, mi, ri);
}
