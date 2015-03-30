/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/relaxation_block.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/block_vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/output.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>

#include <deal.II/base/flow_function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>

#include <amandus.h>

using namespace dealii;


template <int dim>
AmandusUMFPACK<dim>::AmandusUMFPACK(
  Triangulation<dim>& triangulation,
  const FiniteElement<dim>& fe)
		:
		AmandusApplicationSparse<dim>(triangulation, fe, true)
{}


//----------------------------------------------------------------------//

template <int dim>
AmandusResidual<dim>::AmandusResidual(const AmandusApplicationSparse<dim>& application,
				      AmandusIntegrator<dim>& integrator)
		:
		application(&application),
		integrator(&integrator)
{}


template <int dim>
void
AmandusResidual<dim>::operator() (dealii::AnyData &out, const dealii::AnyData &in)
{
  integrator->extract_data(in);
  *out.entry<Vector<double>*>(0) = 0.;
  application->assemble_right_hand_side(out, in, *integrator);
  
  // if we are assembling the residual for a Newton step within a
  // timestepping scheme, we have to subtract the previous time.
  const Vector<double>* p = in.try_read_ptr<Vector<double> >("Previous time");
  if (p != 0)
    out.entry<Vector<double>*>(0)->add(-1., *p);
}


//----------------------------------------------------------------------//

template <int dim>
AmandusSolve<dim>::AmandusSolve(AmandusApplicationSparse<dim>& application,
				AmandusIntegrator<dim>& integrator)
		:
		application(&application),
		integrator(&integrator)
{}


template <int dim>
void
AmandusSolve<dim>::operator() (dealii::AnyData &out, const dealii::AnyData &in)
{
  integrator->extract_data(in);  
  if (this->notifications.test(Algorithms::Events::initial)
      || this->notifications.test(Algorithms::Events::remesh)
      || this->notifications.test(Algorithms::Events::bad_derivative))
    {
      dealii::deallog << "Assemble matrices" << std::endl;
      application->assemble_matrix(in, *integrator);
      application->assemble_mg_matrix(in, *integrator);
      this->notifications.clear();
    }
  const Vector<double>* rhs = in.read_ptr<Vector<double> >(0);
  Vector<double>* solution = out.entry<Vector<double>*>(0);
  
  application->solve(*solution, *rhs);
//  deallog << "Norms " << rhs->l2_norm() << ' ' << solution->l2_norm() << std::endl;
}


template class AmandusUMFPACK<2>;
template class AmandusUMFPACK<3>;

template class AmandusResidual<2>;
template class AmandusResidual<3>;

template class AmandusSolve<2>;
template class AmandusSolve<3>;
