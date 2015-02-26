/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
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

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>

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

#include <amandus.h>

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
{
  deallog << "Finite element: " << fe.get_name() << std::endl;
}

template <int dim>
void
AmandusApplicationSparse<dim>::parse_parameters(dealii::ParameterHandler &param)
{
  param.enter_subsection("Linear Solver");
  control.parse_parameters(param);
  param.leave_subsection();
  
  this->param = &param;
}



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
  dof_handler.distribute_dofs(*this->fe);
  this->dof_handler.distribute_mg_dofs(*this->fe);
  dof_handler.initialize_local_block_info();
  unsigned int n_dofs = dof_handler.n_dofs();
  
  deallog << "DoFHandler: " << this->dof_handler.n_dofs()
	  << std::endl;

  setup_constraints ();

  CompressedSparsityPattern c_sparsity(n_dofs);
  DoFTools::make_flux_sparsity_pattern(dof_handler, c_sparsity, constraints());
  sparsity.copy_from(c_sparsity);
  matrix.reinit(sparsity);  
}


template <int dim>
void
AmandusApplicationSparse<dim>::set_boundary(unsigned int index, dealii::ComponentMask mask)
{
  if (boundary_masks.size() <= index)
    boundary_masks.resize(index+1);
  boundary_masks[index] = mask;
}


template <int dim>
void AmandusApplicationSparse<dim>::setup_constraints()
{
  hanging_node_constraints.clear();
  DoFTools::make_hanging_node_constraints(this->dof_handler, this->hanging_node_constraints);
  hanging_node_constraints.close();
  deallog << "Hanging nodes " << hanging_node_constraints.n_constraints() << std::endl;
  
  constraint_matrix.clear();
  for (unsigned int i=0;i<boundary_masks.size();++i)
    DoFTools::make_zero_boundary_constraints(this->dof_handler, i, this->constraint_matrix, boundary_masks[i]);
  DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraint_matrix);
  constraint_matrix.close();
  deallog << "Constrained " << constraint_matrix.n_constraints() << " dofs" << std::endl;
}


template <int dim>
void
AmandusApplicationSparse<dim>::assemble_matrix(
  const dealii::AnyData &in,
  const AmandusIntegrator<dim>& integrator)
{
  matrix = 0.;

  MeshWorker::IntegrationInfoBox<dim> info_box;
  for (typename std::vector<std::string>::const_iterator i=integrator.input_vector_names.begin();
       i != integrator.input_vector_names.end();++i)
    {
      info_box.cell_selector.add(*i, true, true, false);
    }
  UpdateFlags update_flags = integrator.update_flags();
 
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*fe, mapping, in, Vector<double>(),
			&dof_handler.block_info());

  MeshWorker::DoFInfo<dim> dof_info(dof_handler.block_info());

  MeshWorker::Assembler::MatrixSimple<SparseMatrix<double> > assembler;
  assembler.initialize_local_blocks(dof_handler.block_info().local());
  assembler.initialize(matrix);
  assembler.initialize(constraints());

  MeshWorker::LoopControl control;
  control.cells_first = false;
  MeshWorker::integration_loop<dim, dim>(
    dof_handler.begin_active(), dof_handler.end(),
    dof_info, info_box, integrator, assembler, control);
  
  for (unsigned int i=0;i<matrix.m();++i)
    if (constraints().is_constrained(i))
      matrix.diag_element(i) = 1.;

  if (use_umfpack)
    {
      inverse.initialize(matrix);
    }
}


template <int dim>
void
AmandusApplicationSparse<dim>::assemble_mg_matrix(
  const dealii::AnyData &,
  const AmandusIntegrator<dim>&)
{
}


template <int dim>
void
AmandusApplicationSparse<dim>::assemble_right_hand_side(
  AnyData &out,
  const AnyData &in,
  const AmandusIntegrator<dim>& integrator) const
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
  info_box.initialize(*this->fe, this->mapping, in, Vector<double>(),
		      &dof_handler.block_info());
  
  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());

  MeshWorker::Assembler::ResidualSimple<Vector<double> > assembler;  
  assembler.initialize(this->constraints());
  assembler.initialize(out);
  
  MeshWorker::LoopControl control;
  control.cells_first = false;
  MeshWorker::integration_loop<dim, dim>(
    this->dof_handler.begin_active(), this->dof_handler.end(),
    dof_info, info_box,
    integrator, assembler, control);
}


template <int dim>
void
AmandusApplicationSparse<dim>::verify_residual(
  AnyData &out,
  const AnyData &in,
  const AmandusIntegrator<dim>& integrator) const
{
  MeshWorker::IntegrationInfoBox<dim> info_box;
  for (typename std::vector<std::string>::const_iterator i=integrator.input_vector_names.begin();
       i != integrator.input_vector_names.end();++i)
    {
      info_box.cell_selector.add(*i, true, true, false);
      info_box.boundary_selector.add(*i, true, true, false);
      info_box.face_selector.add(*i, true, true, false);
    }
  
  UpdateFlags update_flags = integrator.update_flags();
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*this->fe, this->mapping, in, Vector<double>(),
		      &dof_handler.block_info());
  
  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());

  MeshWorker::Assembler::ResidualSimple<Vector<double> > assembler;  
  assembler.initialize(this->constraint_matrix);
  assembler.initialize(out);
  
  MeshWorker::LoopControl control;
  control.cells_first = false;
  MeshWorker::integration_loop<dim, dim>(
    this->dof_handler.begin_active(), this->dof_handler.end(),
    dof_info, info_box,
    integrator, assembler, control);
  (*out.entry<Vector<double>*>(0)) *= -1.;

  const Vector<double>* p = in.try_read_ptr<Vector<double> >("Newton iterate");
  matrix.vmult_add(*out.entry<Vector<double>*>(0), *p);
}


template <int dim>
void
AmandusApplicationSparse<dim>::solve(Vector<double>& sol, const Vector<double>& rhs)
{
  SolverGMRES<Vector<double> >::AdditionalData solver_data(40, true);
  SolverGMRES<Vector<double> > solver(control, solver_data);

  PreconditionIdentity identity;
  if (use_umfpack)
    solver.solve(matrix, sol, rhs, this->inverse);
  else
    solver.solve(matrix, sol, rhs, identity);
  constraints().distribute(sol);
}


////
template <int dim>
double AmandusApplicationSparse<dim>::estimate(
  const AnyData &in,
  const AmandusIntegrator<dim>& integrator)
{
  estimates.block(0).reinit(triangulation->n_active_cells());
  unsigned int i=0;
  for(typename Triangulation<dim>::active_cell_iterator cell=triangulation->begin_active();
      cell != triangulation->end() ; ++cell, ++i)
    cell->set_user_index(i);
  MeshWorker::IntegrationInfoBox<dim> info_box;
  
  //TODO: choice of quadrature rule needs to be adjusted. E.g. the estimator for
  //Darcy's equation integrates a postprocessed solution of higher degree
  //than the original solution, thus we need a higher order quadrature
  //formula to obtain correct results
  const unsigned int n_gauss_points= dof_handler.get_fe().tensor_degree()+4;
  info_box.initialize_gauss_quadrature(n_gauss_points, n_gauss_points+1, n_gauss_points) ;
  
  info_box.cell_selector.add("solution", true, true,true);
  info_box.face_selector.add("solution",true,true,true);
  info_box.boundary_selector.add("solution", true, true, false);
  UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients | update_hessians;
  info_box.add_update_flags_all(update_flags);
  
  info_box.initialize(*fe, mapping, in, Vector<double>());
  
  MeshWorker::DoFInfo<dim> dof_info(dof_handler);
  
  MeshWorker::Assembler::CellsAndFaces<double> assembler ;
  AnyData out_data;
  BlockVector<double>* est =&estimates ;
  out_data.add (est,"cells" ); 
  
  assembler.initialize(out_data, false);
  
  MeshWorker::LoopControl control;
  control.cells_first = false;
  MeshWorker::integration_loop< dim , dim >(
    dof_handler.begin_active(), dof_handler.end(),
    dof_info, info_box,
    integrator, assembler, control);
  
  return estimates.block(0).l2_norm();
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
AmandusApplicationSparse<dim>::error(
  BlockVector<double>& errors,
  const dealii::AnyData &solution_data,
  const AmandusIntegrator<dim>& integrator)
{
  for (unsigned int i=0;i<errors.n_blocks();++i)
    errors.block(i).reinit(triangulation->n_active_cells());
  errors.collect_sizes();

  unsigned int i=0;
  for (typename Triangulation<dim>::active_cell_iterator cell = triangulation->begin_active();
       cell != triangulation->end(); ++cell,++i)
    cell->set_user_index(i);

  MeshWorker::IntegrationInfoBox<dim> info_box;
  info_box.cell_selector.add("solution", true, true, false);
  info_box.boundary_selector.add("solution", true, false, false);
  info_box.face_selector.add("solution", true, false, false);
  const unsigned int degree = this->fe->tensor_degree();
  info_box.initialize_gauss_quadrature(degree+2, degree+2, degree+2);
  
  UpdateFlags update_flags = integrator.update_flags();
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*this->fe, this->mapping, solution_data, Vector<double>(),
		      &this->dof_handler.block_info());
  
  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());
  
  MeshWorker::Assembler::CellsAndFaces<double> assembler;
  AnyData out_data;
  BlockVector<double> *est = &errors;
  out_data.add(est, "cells");
  assembler.initialize(out_data, false);
  
  MeshWorker::LoopControl control;
  control.cells_first = false;
  MeshWorker::integration_loop<dim, dim> (
    dof_handler.begin_active(), dof_handler.end(),
    dof_info, info_box,
    integrator, assembler, control);
}

template <int dim>
void
AmandusApplicationSparse<dim>::error(
  BlockVector<double>& errors,
  const dealii::AnyData &solution_data,
  const ErrorIntegrator<dim>& integrator)
{
  //TODO: avoid duplication
  errors.reinit(integrator.size(), triangulation->n_active_cells());
  errors.collect_sizes();

  unsigned int i=0;
  typedef typename Triangulation<dim>::active_cell_iterator cell_it;
  for (cell_it cell = triangulation->begin_active();
       cell != triangulation->end(); ++cell,++i)
  {
    cell->set_user_index(i);
  }

  MeshWorker::IntegrationInfoBox<dim> info_box;
  info_box.cell_selector.add("solution", true, true, false);
  info_box.boundary_selector.add("solution", true, false, false);
  info_box.face_selector.add("solution", true, false, false);
  const unsigned int degree = this->fe->tensor_degree();
  info_box.initialize_gauss_quadrature(degree+2, degree+2, degree+2);
  
  UpdateFlags update_flags = integrator.update_flags();
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*this->fe, this->mapping, solution_data, Vector<double>(),
		      &this->dof_handler.block_info());
  
  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());
  
  MeshWorker::Assembler::CellsAndFaces<double> assembler;
  AnyData out_data;
  BlockVector<double> *est = &errors;
  out_data.add(est, "cells");
  assembler.initialize(out_data, false);
  
  MeshWorker::LoopControl control;
  control.cells_first = false;
  MeshWorker::integration_loop<dim, dim> (
    dof_handler.begin_active(), dof_handler.end(),
    dof_info, info_box,
    integrator, assembler, control);
}


template <int dim>
void
AmandusApplicationSparse<dim>::error(
  const dealii::AnyData &solution_data,
  const AmandusIntegrator<dim>& integrator,
  unsigned int num_errs)
{
  BlockVector<double> errors(num_errs);
  error(errors, solution_data, integrator);
  
  for (unsigned int i=0;i<num_errs;++i)
    deallog << "Error(" << i << "): " << errors.block(i).l2_norm() << std::endl;
}


template <int dim>
void AmandusApplicationSparse<dim>::output_results (const unsigned int cycle,
						    const AnyData* in) const
{
  DataOut<dim> data_out;
  if(param != 0)
  {
    param->enter_subsection("Output");
    data_out.parse_parameters(*param);
    param->leave_subsection();
  } else {
    data_out.set_default_format(DataOutBase::vtk);
  }

  data_out.attach_dof_handler(dof_handler);
  if (in != 0)
  {
    for (unsigned int i=0;i<in->size();++i)
      data_out.add_data_vector(*(in->entry<Vector<double>*>(i)), in->name(i));
  }
  else
  {    
    AssertThrow(false, ExcNotImplemented());
  }
  data_out.build_patches (this->fe->tensor_degree());

  std::ostringstream filename;
  filename << "solution-"
    << cycle
    << data_out.default_suffix();

  std::ofstream output(filename.str().c_str());
  data_out.write(output);
}

template class AmandusApplicationSparse<2>;
template class AmandusApplicationSparse<3>;
