/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/relaxation_block.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_list.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/multigrid/mg_dof_handler.h>

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
AmandusApplication<dim>::AmandusApplication(
  Triangulation<dim>& triangulation,
  const FiniteElement<dim>& fe)
		:
  AmandusApplicationSparseMultigrid<dim>(triangulation, fe)
{}


template <int dim>
void AmandusApplication<dim>::setup_constraints()
{
  this->constraints.clear();

  typename FunctionMap<dim>::type homogen_bc;
  ZeroFunction<dim> zero_function (dim);
  homogen_bc[0] = &zero_function;

  DoFTools::make_zero_boundary_constraints(this->dof_handler, this->constraints);
  this->constraints.close();

  this->mg_constraints.clear();
  this->mg_constraints.initialize(this->dof_handler, homogen_bc);
}


//----------------------------------------------------------------------//


template <int dim>
AmandusUMFPACK<dim>::AmandusUMFPACK(
  Triangulation<dim>& triangulation,
  const FiniteElement<dim>& fe)
		:
		AmandusApplicationSparse<dim>(triangulation, fe, true)
{}


template <int dim>
void AmandusUMFPACK<dim>::setup_constraints()
{
  this->constraints.clear();
  DoFTools::make_zero_boundary_constraints(this->dof_handler, this->constraints);
  this->constraints.close();
}


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
  const double* timestep = in.try_read_ptr<double>("Timestep");
  if (timestep != 0)
    {
      integrator->timestep = *timestep;
//      deallog << "Explicit timestep " << integrator->timestep << std::endl;
    }
  
  *out.entry<Vector<double>*>(0) = 0.;
  application->assemble_right_hand_side(out, in, *integrator);
  
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
  const double* timestep = in.try_read_ptr<double>("Timestep");  
  if (timestep != 0)
    {
      integrator->timestep = *timestep;
//      deallog << "Implicit timestep " << integrator->timestep << std::endl;
    }
  
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


template class AmandusApplication<2>;
template class AmandusApplication<3>;

template class AmandusUMFPACK<2>;
template class AmandusUMFPACK<3>;

template class AmandusResidual<2>;
template class AmandusResidual<3>;

template class AmandusSolve<2>;
template class AmandusSolve<3>;
