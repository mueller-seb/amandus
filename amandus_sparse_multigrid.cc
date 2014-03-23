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

#include "amandus.h"

using namespace dealii;

template <int dim>
AmandusApplicationSparseMultigrid<dim>::AmandusApplicationSparseMultigrid(
  Triangulation<dim>& triangulation,
  const FiniteElement<dim>& fe)
		:
  AmandusApplicationSparse<dim>(triangulation, fe),
  mg_transfer(this->constraints, mg_constraints)
{}


template <int dim>
void
AmandusApplicationSparseMultigrid<dim>::setup_system()
{
  this->dof_handler.distribute_mg_dofs(*this->fe);
  mg_transfer.clear();
  AmandusApplicationSparse<dim>::setup_system();
  
  mg_transfer.initialize_constraints(this->constraints, mg_constraints);
  mg_transfer.build_matrices(this->dof_handler);
  
  const unsigned int n_levels = this->triangulation->n_levels();
  mg_matrix.resize(0, n_levels-1);
  mg_matrix.clear();
  mg_matrix_up.resize(0, n_levels-1);
  mg_matrix_up.clear();
  mg_matrix_down.resize(0, n_levels-1);
  mg_matrix_down.clear();
  mg_sparsity.resize(0, n_levels-1);
  
  for (unsigned int level=mg_sparsity.min_level();
       level<=mg_sparsity.max_level();++level)
    {
      CompressedSparsityPattern c_sparsity(this->dof_handler.n_dofs(level));      
      MGTools::make_flux_sparsity_pattern(this->dof_handler, c_sparsity, level);
      mg_sparsity[level].copy_from(c_sparsity);
      mg_matrix[level].reinit(mg_sparsity[level]);
      mg_matrix_up[level].reinit(mg_sparsity[level]);
      mg_matrix_down[level].reinit(mg_sparsity[level]);
    }
}


template <int dim>
void AmandusApplicationSparseMultigrid<dim>::setup_constraints()
{
  this->constraints.clear();
  this->constraints.close();
  
  this->mg_constraints.clear();
  this->mg_constraints.initialize(this->dof_handler);
}


template <int dim>
void
AmandusApplicationSparseMultigrid<dim>::assemble_mg_matrix(const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
					    const dealii::NamedData<dealii::Vector<double> *> &in)
{
  mg_matrix = 0.;
  std::vector<MGLevelObject<Vector<double> > > aux(in.size());
  const NamedData<MGLevelObject<Vector<double> > *> mg_in;
  unsigned int k=0;
  // for (typename std::vector<std::string>::const_iterator i=integrator.input_vector_names.begin();
  //      i != integrator.input_vector_names.end();++i)
  //   {
  //     info_box.cell_selector.add(*i, true, true, false);
  //   }
  
    
  MeshWorker::IntegrationInfoBox<dim> info_box;
  UpdateFlags update_flags = update_values | update_gradients | update_hessians;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*this->fe, this->mapping, &this->dof_handler.block_info());

  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());
  
  MeshWorker::Assembler::MGMatrixSimple<SparseMatrix<double> > assembler;
  assembler.initialize(mg_constraints);
  assembler.initialize_local_blocks(this->dof_handler.block_info().local());
  assembler.initialize(mg_matrix);
  assembler.initialize_interfaces(mg_matrix_up, mg_matrix_down);

  MeshWorker::integration_loop<dim, dim> (
    this->dof_handler.begin_mg(), this->dof_handler.end_mg(),
    dof_info, info_box, integrator, assembler);
}


template <int dim>
void
AmandusApplicationSparseMultigrid<dim>::solve(Vector<double>& sol, const Vector<double>& rhs)
{
  const unsigned int minlevel = 0;
  SolverGMRES<Vector<double> >::AdditionalData solver_data(40, true);
  SolverGMRES<Vector<double> > solver(this->control, solver_data);
  // SolverCG<Vector<double> >::AdditionalData solver_data(false, false, true, true);
  // SolverCG<Vector<double> > solver(control, solver_data);
  // SolverRichardson<Vector<double> > solver(control, .6);
  
  FullMatrix<double> coarse_matrix;
  coarse_matrix.copy_from (mg_matrix[minlevel]);
  MGCoarseGridSVD<double, Vector<double> > mg_coarse;
  mg_coarse.initialize(coarse_matrix, 1.e-15);
  
  typedef RelaxationBlockSSOR<SparseMatrix<double> > RELAXATION;
  MGLevelObject<RELAXATION::AdditionalData> smoother_data(minlevel, mg_matrix.max_level());  
  mg::SmootherRelaxation<RELAXATION, Vector<double> > mg_smoother;

  for (unsigned int l=minlevel+1;l<=smoother_data.max_level();++l)
    {
      DoFTools::make_vertex_patches(smoother_data[l].block_list, this->dof_handler, l, true);
      smoother_data[l].block_list.compress();
      smoother_data[l].relaxation = 1.;
      smoother_data[l].inversion = PreconditionBlockBase<double>::svd;
      smoother_data[l].threshold = 1.e-12;
    } 
  mg_smoother.initialize(mg_matrix, smoother_data);
  if (false)
    for (unsigned int l=minlevel+1;l<=smoother_data.max_level();++l)
      {
	deallog << "Level " << l << ' ';
	mg_smoother[l].log_statistics();
      }
  
  mg_smoother.set_steps(1);
  mg_smoother.set_variable(false);
  
  MGMatrix<SparseMatrix<double>, Vector<double> > mgmatrix(&mg_matrix);
  MGMatrix<SparseMatrix<double>, Vector<double> > mgdown(&mg_matrix_down);
  MGMatrix<SparseMatrix<double>, Vector<double> > mgup(&mg_matrix_up);
  
  Multigrid<Vector<double> > mg(this->dof_handler, mgmatrix,
				mg_coarse, mg_transfer,
				mg_smoother, mg_smoother);
  mg.set_edge_matrices(mgdown, mgup);
  mg.set_minlevel(minlevel);
  
  PreconditionMG<dim, Vector<double>,
    MGTransferPrebuilt<Vector<double> > >
    preconditioner(this->dof_handler, mg, mg_transfer);
  try 
    {
      solver.solve(this->matrix, sol, rhs, preconditioner);
    }
  catch(...) {}
  this->constraints.distribute(sol);
}




template class AmandusApplicationSparseMultigrid<2>;
template class AmandusApplicationSparseMultigrid<3>;

