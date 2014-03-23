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
AmandusApplicationSparse<dim>::AmandusApplicationSparse(
  Triangulation<dim>& triangulation,
  const FiniteElement<dim>& fe,
  bool use_umfpack)
		:
		control(100, 1.e-20, 1.e-2),
		triangulation(&triangulation),
		fe(&fe),
		dof_handler(triangulation),
		use_umfpack(use_umfpack),
	        estimates(1)		
{}


template <int dim>
void
AmandusApplicationSparse<dim>::setup_vector(Vector<double>& v) const
{
  v.reinit(dof_handler.n_dofs());
}


template <int dim>
void
AmandusApplicationSparse<dim>::setup_system()
{
  dof_handler.distribute_dofs(*fe);
  dof_handler.initialize_local_block_info();
  unsigned int n_dofs = dof_handler.n_dofs();
  
  deallog << "DoFHandler " << this->dof_handler.n_dofs() << " dofs, level dofs";
  for (unsigned int l=0;l<this->triangulation->n_levels();++l)
    deallog << ' ' << this->dof_handler.n_dofs(l);
  deallog << std::endl;

  setup_constraints ();

  CompressedSparsityPattern c_sparsity(n_dofs);
  DoFTools::make_flux_sparsity_pattern(dof_handler, c_sparsity, constraints);
  sparsity.copy_from(c_sparsity);
  matrix.reinit(sparsity);  
}


template <int dim>
void AmandusApplicationSparse<dim>::setup_constraints()
{
  constraints.clear();
  constraints.close();
}


template <int dim>
void
AmandusApplicationSparse<dim>::assemble_matrix(const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
					 const dealii::NamedData<dealii::Vector<double> *> &in)
{
  matrix = 0.;

  MeshWorker::IntegrationInfoBox<dim> info_box;
  for (typename std::vector<std::string>::const_iterator i=integrator.input_vector_names.begin();
       i != integrator.input_vector_names.end();++i)
    {
      info_box.cell_selector.add(*i, true, true, false);
    }
  UpdateFlags update_flags = update_values | update_gradients | update_hessians | update_quadrature_points;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*fe, mapping, in, &dof_handler.block_info());

  MeshWorker::DoFInfo<dim> dof_info(dof_handler.block_info());

  MeshWorker::Assembler::MatrixSimple<SparseMatrix<double> > assembler;
  assembler.initialize_local_blocks(dof_handler.block_info().local());
  assembler.initialize(matrix);
  assembler.initialize(constraints);

  MeshWorker::integration_loop<dim, dim>(
    dof_handler.begin_active(), dof_handler.end(),
    dof_info, info_box, integrator, assembler);
  
  for (unsigned int i=0;i<matrix.m();++i)
    if (constraints.is_constrained(i))
      matrix.diag_element(i) = 1.;

  if (use_umfpack)
    {
      inverse.initialize(matrix);
    }
}


template <int dim>
void
AmandusApplicationSparse<dim>::assemble_mg_matrix(
  const dealii::MeshWorker::LocalIntegrator<dim>&,
  const dealii::NamedData<dealii::Vector<double> *> &)
{
}


template <int dim>
void
AmandusApplicationSparse<dim>::assemble_right_hand_side(
  const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
  NamedData<Vector<double> *> &out,
  const NamedData<Vector<double> *> &in) const
{
  MeshWorker::IntegrationInfoBox<dim> info_box;
  for (typename std::vector<std::string>::const_iterator i=integrator.input_vector_names.begin();
       i != integrator.input_vector_names.end();++i)
    {
      info_box.cell_selector.add(*i, true, true, false);
      info_box.boundary_selector.add(*i, true, true, false);
      info_box.face_selector.add(*i, true, true, false);
    }
  
  UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*this->fe, this->mapping, in, &dof_handler.block_info());
  
  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());

  MeshWorker::Assembler::ResidualSimple<Vector<double> > assembler;  
  assembler.initialize(this->constraints);
  assembler.initialize(out);
  
  MeshWorker::integration_loop<dim, dim>(
    this->dof_handler.begin_active(), this->dof_handler.end(),
    dof_info, info_box,
    integrator, assembler);
}


template <int dim>
void
AmandusApplicationSparse<dim>::verify_residual(const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
					 NamedData<Vector<double> *> &out,
					 const NamedData<Vector<double> *> &in) const
{
  MeshWorker::IntegrationInfoBox<dim> info_box;
  for (typename std::vector<std::string>::const_iterator i=integrator.input_vector_names.begin();
       i != integrator.input_vector_names.end();++i)
    {
      info_box.cell_selector.add(*i, true, true, false);
      info_box.boundary_selector.add(*i, true, true, false);
      info_box.face_selector.add(*i, true, true, false);
    }
  
  UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*this->fe, this->mapping, in, &dof_handler.block_info());
  
  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());

  MeshWorker::Assembler::ResidualSimple<Vector<double> > assembler;  
  assembler.initialize(this->constraints);
  assembler.initialize(out);
  
  MeshWorker::integration_loop<dim, dim>(
    this->dof_handler.begin_active(), this->dof_handler.end(),
    dof_info, info_box,
    integrator, assembler);
  (*out(0)) *= -1.;

  matrix.vmult_add(*out(0), *in(0));
}


template <int dim>
void
AmandusApplicationSparse<dim>::solve(Vector<double>& sol, const Vector<double>& rhs)
{
  SolverGMRES<Vector<double> >::AdditionalData solver_data(40, true);
  SolverGMRES<Vector<double> > solver(control, solver_data);

  PreconditionIdentity preconditioner;
  try 
    {
      solver.solve(matrix, sol, rhs, preconditioner);
    }
  catch(...) {}
  constraints.distribute(sol);
}


////
template <int dim>
double AmandusApplicationSparse<dim>::estimate(const MeshWorker::LocalIntegrator<dim>& integrator)
{
  // estimates.block(0).reinit(triangulation->n_active_cells());
  // unsigned int i=0;
  // for(typename Triangulation<dim>::active_cell_iterator cell=triangulation->begin_active();
  //     cell != triangulation->end() ; ++cell, ++i)
  //   cell->set_user_index(i);
  // MeshWorker::IntegrationInfoBox<dim> info_box;
  
  // const unsigned int n_gauss_points= dof_handler.get_fe().tensor_degree()+1;
  // info_box.initialize_gauss_quadrature(n_gauss_points, n_gauss_points+1, n_gauss_points) ;
  
  // NamedData<Vector<double>*> solution_data ;
  // solution_data.add(&solution, "solution");
  
  // info_box.cell_selector.add("solution", false, true,true);
  // info_box.face_selector.add("solution",true,true,true);
  // info_box.boundary_selector.add("solution", true, true, false);
  // UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients | update_hessians;
  // info_box.add_update_flags_all(update_flags);
  
  // info_box.initialize(*fe, mapping, solution_data);
  
  // MeshWorker::DoFInfo<dim> dof_info(dof_handler);
  
  // MeshWorker::Assembler::CellsAndFaces<double> assembler ;
  // NamedData< BlockVector<double>* > out_data;
  // BlockVector<double>* est =&estimates ;
  // out_data.add (est,"cells" ); 
  
  // assembler.initialize(out_data, false);
  
  // MeshWorker::integration_loop< dim , dim >(
  //   dof_handler.begin_active(), dof_handler.end(),
  //   dof_info, info_box,
  //   integrator, assembler);
  
  // return estimates.block(0).l2_norm();
}


////
template <int dim>
void AmandusApplicationSparse<dim>::refine_mesh (const bool global)
{
  bool cell_refined = false;
  if (global || !cell_refined)
    triangulation->refine_global(1);
  else
    triangulation->execute_coarsening_and_refinement ();
  
  deallog << "Triangulation "
	  << triangulation->n_active_cells() << " cells, "
	  << triangulation->n_levels() << " levels" << std::endl;
}


template <int dim>
void
AmandusApplicationSparse<dim>::error(const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
const dealii::NamedData<dealii::Vector<double> *> &solution_data,
unsigned int num_errs)
{
  BlockVector<double> errors(num_errs);
  for (unsigned int i=0;i<num_errs;++i)
    errors.block(i).reinit(triangulation->n_active_cells());
  unsigned int i=0;
  for (typename Triangulation<dim>::active_cell_iterator cell = triangulation->begin_active();
       cell != triangulation->end(); ++cell,++i)
    cell->set_user_index(i);

  MeshWorker::IntegrationInfoBox<dim> info_box;
  info_box.cell_selector.add("solution", true, true, false);
  info_box.boundary_selector.add("solution", true, false, false);
  info_box.face_selector.add("solution", true, false, false);
  
  UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*this->fe, this->mapping, solution_data, &this->dof_handler.block_info());
  
  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());
  
  MeshWorker::Assembler::CellsAndFaces<double> assembler;
  NamedData<BlockVector<double>* > out_data;
  BlockVector<double> *est = &errors;
  out_data.add(est, "cells");
  assembler.initialize(out_data, false);
  
  MeshWorker::integration_loop<dim, dim> (
    dof_handler.begin_active(), dof_handler.end(),
    dof_info, info_box,
    integrator, assembler);

  for (unsigned int i=0;i<num_errs;++i)
    deallog << "Error(" << i << "): " << errors.block(i).l2_norm() << std::endl;
}



template <int dim>
void AmandusApplicationSparse<dim>::output_results (const unsigned int cycle,
						    const NamedData<Vector<double>*>* in) const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  if (in != 0)
    {
      for (unsigned int i=0;i<in->size();++i)
	data_out.add_data_vector(*((*in)(i)), in->name(i));
    }
  else
    {    
      AssertThrow(false, ExcNotImplemented());
    }
  data_out.build_patches (3);

  std::ostringstream filename;
  filename << "solution-"
    << cycle
    << ".gpl";

  std::ofstream output (filename.str().c_str());
  data_out.write_gnuplot (output);

  // std::ostringstream filename;
  // filename << "solution-"
  //   << cycle
  //   << ".svg";

  // DataOutBase::SvgFlags svg_flags;;
  // svg_flags.height = 400;
  // svg_flags.polar_angle=30;
  
  // data_out.set_flags(svg_flags);
  
  // std::ofstream output (filename.str().c_str());
  // data_out.write_svg (output);
}

template class AmandusApplicationSparse<2>;
template class AmandusApplicationSparse<3>;
