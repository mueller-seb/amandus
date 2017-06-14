/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#include <deal.II/lac/arpack_solver.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/iterative_inverse.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/relaxation_block.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/output.h>
#include <deal.II/meshworker/simple.h>

#include <deal.II/base/flow_function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include <amandus/amandus.h>
#include <amandus/integrator.h>

using namespace dealii;

template <int dim>
void
make_meanvalue_constraints(const DoFHandler<dim>& dofh, const Mapping<dim>& mapping,
                           const ConstraintMatrix& hanging_nodes, ConstraintMatrix& constraints,
                           const ComponentMask& mask)
{
  if (mask.n_selected_components(dofh.get_fe().n_components()) == 0)
  {
    return;
  }
  MeshWorker::IntegrationInfoBox<dim> info_box;
  UpdateFlags update_flags = update_values;
  info_box.add_update_flags_cell(update_flags);
  info_box.initialize(dofh.get_fe(), mapping, &dofh.block_info());

  MeshWorker::DoFInfo<dim> dofinfo(dofh.block_info());

  AnyData out;
  Vector<double> mean_values(dofh.n_dofs());
  out.add(&mean_values, "mean");

  MeshWorker::Assembler::ResidualSimple<Vector<double>> assembler;
  assembler.initialize(hanging_nodes);
  assembler.initialize(out);

  Integrators::MeanIntegrator<dim> integrator;

  MeshWorker::integration_loop<dim, dim>(
    dofh.begin_active(), dofh.end(), dofinfo, info_box, integrator, assembler);

  const FiniteElement<dim>& fe_sys = dofh.get_fe();
  std::vector<bool> component_dof_indices(dofh.n_dofs());
  for (unsigned int component = 0; component < fe_sys.n_components(); ++component)
  {
    if (mask[component])
    {
      ComponentMask current_component(fe_sys.n_components(), false);
      current_component.set(component, true);
      DoFTools::extract_dofs(dofh, current_component, component_dof_indices);
      unsigned int first_dof_idx =
        std::distance(component_dof_indices.begin(),
                      std::find(component_dof_indices.begin(), component_dof_indices.end(), true));
      constraints.add_line(first_dof_idx);
      for (unsigned int dof_idx = first_dof_idx + 1; dof_idx < dofh.n_dofs(); ++dof_idx)
      {
        if (component_dof_indices[dof_idx])
        {
          double coefficient = (-1.0 * mean_values[dof_idx] / mean_values[first_dof_idx]);
          constraints.add_entry(first_dof_idx, dof_idx, coefficient);
        }
      }
    }
  }
}

template <int dim>
AmandusApplicationSparse<dim>::AmandusApplicationSparse(Triangulation<dim>& triangulation,
                                                        const FiniteElement<dim>& fe,
                                                        bool use_umfpack)
  : control(100, 1.e-20, 1.e-2)
  , signals(triangulation.signals)
  , triangulation(&triangulation)
  , fe(&fe)
  , dof_handler(triangulation)
  , meanvalue_mask(fe.n_components(), false)
  , matrix(1)
  , use_umfpack(use_umfpack)
  , estimates(1)
  , output_data_types(fe.n_components())
{
  deallog << "Finite element: " << fe.get_name() << std::endl;

  unsigned int comp = 0;
  deallog << "Output types ";
  for (unsigned int i = 0; i < fe.n_base_elements(); ++i)
  {
    const FiniteElement<dim>& base = fe.base_element(i);
    DataComponentInterpretation::DataComponentInterpretation inter =
      DataComponentInterpretation::component_is_scalar;
    if (base.n_components() == dim)
      inter = DataComponentInterpretation::component_is_part_of_vector;
    for (unsigned int j = 0; j < fe.element_multiplicity(i); ++j)
      for (unsigned int k = 0; k < base.n_components(); ++k)
      {
        output_data_types[comp++] = inter;
        deallog << ((base.n_components() == dim) ? 'v' : 's');
      }
  }
  deallog << std::endl;
}

template <int dim>
void
AmandusApplicationSparse<dim>::parse_parameters(dealii::ParameterHandler& new_param)
{
  new_param.enter_subsection("Linear Solver");
  control.parse_parameters(new_param);
  new_param.leave_subsection();

  this->param = &new_param;
}

template <int dim>
void
AmandusApplicationSparse<dim>::setup_vector(Vector<double>& v) const
{
  v.reinit(dof_handler.n_dofs());
}

template <int dim>
void
AmandusApplicationSparse<dim>::update_vector_inhom_boundary(
  Vector<double>& v, const dealii::Function<dim>& inhom_boundary, bool projection) const
{
  const unsigned int n_comp = this->dof_handler.get_fe().n_components();
  dealii::QGauss<dim - 1> q_boundary(this->dof_handler.get_fe().tensor_degree() + 1);
  typename dealii::FunctionMap<dim>::type boundary_functions;
  boundary_functions[0] = &inhom_boundary;
  std::map<dealii::types::global_dof_index, double> boundary_dofs;
  if (projection)
    dealii::VectorTools::project_boundary_values(
      this->dof_handler, boundary_functions, q_boundary, boundary_dofs);
  else
    for (std::map<dealii::types::boundary_id, dealii::ComponentMask>::const_iterator p =
           boundary_masks.begin();
         p != boundary_masks.end();
         ++p)
      if (p->second.n_selected_components(n_comp) != 0)
        dealii::VectorTools::interpolate_boundary_values(
          this->dof_handler, p->first, inhom_boundary, boundary_dofs, p->second);
  for (auto bdry_dof = boundary_dofs.begin(); bdry_dof != boundary_dofs.end(); ++bdry_dof)
    v(bdry_dof->first) = bdry_dof->second;

  hanging_node_constraints.distribute(v);
}

template <int dim>
void
AmandusApplicationSparse<dim>::setup_system()
{
  dof_handler.distribute_dofs(*this->fe);
  this->dof_handler.distribute_mg_dofs(*this->fe);
  dof_handler.initialize_local_block_info();
  unsigned int n_dofs = dof_handler.n_dofs();

  deallog << "DoFHandler: " << this->dof_handler.n_dofs() << std::endl;

  setup_constraints();

  DynamicSparsityPattern c_sparsity(n_dofs);
  DoFTools::make_flux_sparsity_pattern(dof_handler, c_sparsity, constraints());
  sparsity.copy_from(c_sparsity);
  for (unsigned int m = 0; m < matrix.size(); ++m)
    matrix[m].reinit(sparsity);
}

template <int dim>
void
AmandusApplicationSparse<dim>::set_boundary(dealii::types::boundary_id index,
                                            dealii::ComponentMask mask)
{
  boundary_masks.insert(std::pair<dealii::types::boundary_id, dealii::ComponentMask>(index, mask));
}

template <int dim>
void
AmandusApplicationSparse<dim>::set_meanvalue(dealii::ComponentMask mask)
{
  this->meanvalue_mask = mask;
}

template <int dim>
void
AmandusApplicationSparse<dim>::setup_constraints()
{
  hanging_node_constraints.clear();
  DoFTools::make_hanging_node_constraints(this->dof_handler, this->hanging_node_constraints);
  hanging_node_constraints.close();
  deallog << "Hanging nodes " << hanging_node_constraints.n_constraints() << std::endl;
  const unsigned int n_comp = this->dof_handler.get_fe().n_components();

  constraint_matrix.clear();
  for (std::map<dealii::types::boundary_id, dealii::ComponentMask>::const_iterator p =
         boundary_masks.begin();
       p != boundary_masks.end();
       ++p)
    if (p->second.n_selected_components(n_comp) != 0)
      DoFTools::make_zero_boundary_constraints(
        this->dof_handler, p->first, this->constraint_matrix, p->second);
  make_meanvalue_constraints(this->dof_handler,
                             this->mapping,
                             this->hanging_node_constraints,
                             this->constraint_matrix,
                             this->meanvalue_mask);
  DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraint_matrix);
  constraint_matrix.close();
  deallog << "Constrained " << constraint_matrix.n_constraints() << " dofs" << std::endl;
}

template <int dim>
void
AmandusApplicationSparse<dim>::assemble_matrix(const dealii::AnyData& in,
                                               const AmandusIntegrator<dim>& integrator)
{
  for (unsigned int m = 0; m < matrix.size(); ++m)
    matrix[m] = 0.;

  UpdateFlags update_flags = integrator.update_flags();
  bool values_flag = update_flags & update_values;
  bool gradients_flag = update_flags & update_gradients;
  bool hessians_flag = update_flags & update_hessians;

  MeshWorker::IntegrationInfoBox<dim> info_box;
  for (typename std::vector<std::string>::const_iterator i = integrator.input_vector_names.begin();
       i != integrator.input_vector_names.end();
       ++i)
  {
    info_box.cell_selector.add(*i, values_flag, gradients_flag, hessians_flag);
    info_box.boundary_selector.add(*i, values_flag, gradients_flag, hessians_flag);
    info_box.face_selector.add(*i, values_flag, gradients_flag, hessians_flag);
  }

  info_box.add_update_flags_all(update_flags);
  info_box.add_update_flags_boundary(integrator.update_flags_face());
  info_box.add_update_flags_face(integrator.update_flags_face());
  if (integrator.cell_quadrature != 0)
  {
    info_box.cell_quadrature = *(integrator.cell_quadrature);
  }
  if (integrator.boundary_quadrature != 0)
  {
    info_box.boundary_quadrature = *(integrator.boundary_quadrature);
  }
  if (integrator.face_quadrature != 0)
  {
    info_box.face_quadrature = *(integrator.face_quadrature);
  }
  info_box.initialize(*fe, mapping, in, Vector<double>(), &dof_handler.block_info());

  MeshWorker::DoFInfo<dim> dof_info(dof_handler.block_info());

  MeshWorker::Assembler::MatrixSimple<SparseMatrix<double>> assembler;
  assembler.initialize(matrix);
  assembler.initialize(constraints());

  MeshWorker::LoopControl new_control;
  new_control.cells_first = false;
  MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(),
                                         dof_handler.end(),
                                         dof_info,
                                         info_box,
                                         integrator,
                                         assembler,
                                         new_control);

  for (unsigned int m = 0; m < matrix.size(); ++m)
    for (unsigned int i = 0; i < matrix[m].m(); ++i)
      if (constraints().is_constrained(i))
        matrix[m].diag_element(i) = 1.;

  if (use_umfpack)
  {
    inverse.initialize(matrix[0]);
  }
}

template <int dim>
void
AmandusApplicationSparse<dim>::assemble_mg_matrix(const dealii::AnyData&,
                                                  const AmandusIntegrator<dim>&)
{
}

template <int dim>
void
AmandusApplicationSparse<dim>::assemble_right_hand_side(
  AnyData& out, const AnyData& in, const AmandusIntegrator<dim>& integrator) const
{
  UpdateFlags update_flags = integrator.update_flags();
  bool values_flag = update_flags & update_values;
  bool gradients_flag = update_flags & update_gradients;
  bool hessians_flag = update_flags & update_hessians;

  MeshWorker::IntegrationInfoBox<dim> info_box;
  for (typename std::vector<std::string>::const_iterator i = integrator.input_vector_names.begin();
       i != integrator.input_vector_names.end();
       ++i)
  {
    info_box.cell_selector.add(*i, values_flag, gradients_flag, hessians_flag);
    info_box.boundary_selector.add(*i, values_flag, gradients_flag, hessians_flag);
    info_box.face_selector.add(*i, values_flag, gradients_flag, hessians_flag);
  }

  info_box.add_update_flags_all(update_flags);
  info_box.add_update_flags_boundary(integrator.update_flags_face());
  info_box.add_update_flags_face(integrator.update_flags_face());
  // user defined quadrature rules if set
  if (integrator.cell_quadrature != 0)
  {
    info_box.cell_quadrature = *(integrator.cell_quadrature);
  }
  if (integrator.boundary_quadrature != 0)
  {
    info_box.boundary_quadrature = *(integrator.boundary_quadrature);
  }
  if (integrator.face_quadrature != 0)
  {
    info_box.face_quadrature = *(integrator.face_quadrature);
  }
  info_box.initialize(*this->fe, this->mapping, in, Vector<double>(), &dof_handler.block_info());

  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());

  MeshWorker::Assembler::ResidualSimple<Vector<double>> assembler;
  assembler.initialize(this->constraints());
  assembler.initialize(out);

  MeshWorker::LoopControl new_control;
  new_control.cells_first = false;
  MeshWorker::integration_loop<dim, dim>(this->dof_handler.begin_active(),
                                         this->dof_handler.end(),
                                         dof_info,
                                         info_box,
                                         integrator,
                                         assembler,
                                         new_control);
}

template <int dim>
void
AmandusApplicationSparse<dim>::verify_residual(AnyData& out, const AnyData& in,
                                               const AmandusIntegrator<dim>& integrator) const
{
  UpdateFlags update_flags = integrator.update_flags();
  bool values_flag = update_flags & update_values;
  bool gradients_flag = update_flags & update_gradients;
  bool hessians_flag = update_flags & update_hessians;

  MeshWorker::IntegrationInfoBox<dim> info_box;
  for (typename std::vector<std::string>::const_iterator i = integrator.input_vector_names.begin();
       i != integrator.input_vector_names.end();
       ++i)
  {
    info_box.cell_selector.add(*i, values_flag, gradients_flag, hessians_flag);
    info_box.boundary_selector.add(*i, values_flag, gradients_flag, hessians_flag);
    info_box.face_selector.add(*i, values_flag, gradients_flag, hessians_flag);
  }

  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*this->fe, this->mapping, in, Vector<double>(), &dof_handler.block_info());

  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());

  MeshWorker::Assembler::ResidualSimple<Vector<double>> assembler;
  assembler.initialize(this->constraint_matrix);
  assembler.initialize(out);

  MeshWorker::LoopControl new_control;
  new_control.cells_first = false;
  MeshWorker::integration_loop<dim, dim>(this->dof_handler.begin_active(),
                                         this->dof_handler.end(),
                                         dof_info,
                                         info_box,
                                         integrator,
                                         assembler,
                                         new_control);
  (*out.entry<Vector<double>*>(0)) *= -1.;

  const Vector<double>* p = in.try_read_ptr<Vector<double>>("Newton iterate");
  AssertDimension(matrix.size(), 1);
  matrix[0].vmult_add(*out.entry<Vector<double>*>(0), *p);
}

template <int dim>
void
AmandusApplicationSparse<dim>::solve(Vector<double>& sol, const Vector<double>& rhs)
{
  AssertDimension(matrix.size(), 1);
  SolverGMRES<Vector<double>>::AdditionalData solver_data(40, true);
  SolverGMRES<Vector<double>> solver(control, solver_data);

  PreconditionIdentity identity;
  if (use_umfpack)
    solver.solve(matrix[0], sol, rhs, this->inverse);
  else
    solver.solve(matrix[0], sol, rhs, identity);
  constraints().distribute(sol);
}

#ifdef DEAL_II_WITH_ARPACK
template <int dim>
void
AmandusApplicationSparse<dim>::arpack_solve(std::vector<std::complex<double>>& eigenvalues,
                                            std::vector<Vector<double>>& eigenvectors)
{
  size_t min_Arnoldi_vectors = 20;
  bool symmetric = false;
  unsigned int max_steps = 100;
  double tolerance = 1e-10;
  if (this->param != 0)
  {
    this->param->enter_subsection("Arpack");
    min_Arnoldi_vectors = this->param->get_integer("Min Arnoldi vectors");
    symmetric = this->param->get_bool("Symmetric");
    max_steps = this->param->get_integer("Max steps");
    tolerance = this->param->get_double("Tolerance");
    this->param->leave_subsection();
  }

  if (symmetric)
  {
    AssertDimension(eigenvalues.size(), eigenvectors.size());
  }
  else
    AssertDimension(2 * eigenvalues.size(), eigenvectors.size());

  ArpackSolver::AdditionalData arpack_data(
    std::max(min_Arnoldi_vectors, 2 * eigenvalues.size() + 2),
    ArpackSolver::largest_magnitude,
    symmetric);
  dealii::SolverControl arpack_control(max_steps, tolerance);
  ArpackSolver solver(arpack_control, arpack_data);

  std::vector<dealii::Vector<double>> arpack_vectors(eigenvalues.size() + 1);
  for (unsigned int i = 0; i < arpack_vectors.size(); ++i)
    setup_vector(arpack_vectors[i]);

  if (eigenvectors[0].l2_norm() != 0.)
    solver.set_initial_vector(eigenvectors[0]);

  for (unsigned int i = 0; i < matrix[1].m(); ++i)
    if (constraints().is_constrained(i))
      matrix[1].diag_element(i) = 0.;

  if (use_umfpack)
    solver.solve(matrix[0], matrix[1], inverse, eigenvalues, arpack_vectors, eigenvalues.size());
  else
  {
    SolverGMRES<Vector<double>>::AdditionalData solver_data(40, true);
    PreconditionIdentity identity;
    IterativeInverse<Vector<double>> inv;
    inv.initialize(matrix[0], identity);
    inv.solver.set_control(control);
    inv.solver.set_data(solver_data);
    inv.solver.select("gmres");
    solver.solve(matrix[0], matrix[1], inv, eigenvalues, arpack_vectors, eigenvalues.size());
  }
  for (unsigned int i = 0; i < arpack_vectors.size(); ++i)
    constraints().distribute(arpack_vectors[i]);

  for (unsigned int i = 0; i < eigenvalues.size(); ++i)
  {
    eigenvectors[i] = arpack_vectors[i];
    if (std::fabs(eigenvalues[i].imag() > 1.e-12))
    {
      eigenvectors[i + eigenvalues.size()] = arpack_vectors[i + 1];
      if (i + 1 < eigenvalues.size())
      {
        eigenvectors[i + 1] = arpack_vectors[i];
        eigenvectors[i + 1 + eigenvalues.size()] = arpack_vectors[i + 1];
        eigenvectors[i + 1 + eigenvalues.size()] *= -1;
        ++i;
      }
    }
  }
}
#else
template <int dim>
void
AmandusApplicationSparse<dim>::arpack_solve(std::vector<std::complex<double>>& /*eigenvalues*/,
                                            std::vector<Vector<double>>& /*eigenvectors*/)
{
  AssertThrow(false, ExcNeedArpack());
}
#endif

////
template <int dim>
double
AmandusApplicationSparse<dim>::estimate(const AnyData& in, AmandusIntegrator<dim>& integrator)
{
  estimates.block(0).reinit(triangulation->n_active_cells());
  unsigned int i = 0;
  for (typename Triangulation<dim>::active_cell_iterator cell = triangulation->begin_active();
       cell != triangulation->end();
       ++cell, ++i)
    cell->set_user_index(i);
  MeshWorker::IntegrationInfoBox<dim> info_box;

  if (integrator.cell_quadrature != 0)
  {
    info_box.cell_quadrature = *(integrator.cell_quadrature);
  }
  if (integrator.boundary_quadrature != 0)
  {
    info_box.boundary_quadrature = *(integrator.boundary_quadrature);
  }
  if (integrator.face_quadrature != 0)
  {
    info_box.face_quadrature = *(integrator.face_quadrature);
  }

  integrator.extract_data(in);

  UpdateFlags update_flags = integrator.update_flags();
  bool values_flag = update_flags & update_values;
  bool gradients_flag = update_flags & update_gradients;
  bool hessians_flag = update_flags & update_hessians;

  if (integrator.input_vector_names.size() == 0)
  {
    info_box.cell_selector.add("solution", values_flag, gradients_flag, hessians_flag);
    info_box.face_selector.add("solution", values_flag, gradients_flag, hessians_flag);
    info_box.boundary_selector.add("solution", values_flag, gradients_flag, hessians_flag);
  }
  else
  {
    for (typename std::vector<std::string>::const_iterator i =
           integrator.input_vector_names.begin();
         i != integrator.input_vector_names.end();
         ++i)
    {
      info_box.cell_selector.add(*i, values_flag, gradients_flag, hessians_flag);
      info_box.boundary_selector.add(*i, values_flag, gradients_flag, hessians_flag);
      info_box.face_selector.add(*i, values_flag, gradients_flag, hessians_flag);
    }
  }
  info_box.add_update_flags_all(update_flags);

  info_box.initialize(*fe, mapping, in, Vector<double>());

  MeshWorker::DoFInfo<dim> dof_info(dof_handler);

  MeshWorker::Assembler::CellsAndFaces<double> assembler;
  AnyData out_data;
  BlockVector<double>* est = &estimates;
  out_data.add(est, "cells");

  assembler.initialize(out_data, false);

  MeshWorker::LoopControl new_control;
  new_control.cells_first = false;
  MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(),
                                         dof_handler.end(),
                                         dof_info,
                                         info_box,
                                         integrator,
                                         assembler,
                                         new_control);

  return estimates.block(0).l2_norm();
}

////
template <int dim>
void
AmandusApplicationSparse<dim>::refine_mesh(const bool global)
{
  if (global)
    triangulation->refine_global(1);
  else
    triangulation->execute_coarsening_and_refinement();

  deallog << "Triangulation " << triangulation->n_active_cells() << " cells, "
          << triangulation->n_levels() << " levels" << std::endl;
}

template <int dim>
void
AmandusApplicationSparse<dim>::error(BlockVector<double>& errors,
                                     const dealii::AnyData& solution_data,
                                     const AmandusIntegrator<dim>& integrator)
{
  for (unsigned int i = 0; i < errors.n_blocks(); ++i)
    errors.block(i).reinit(triangulation->n_active_cells());
  errors.collect_sizes();

  unsigned int i = 0;
  for (typename Triangulation<dim>::active_cell_iterator cell = triangulation->begin_active();
       cell != triangulation->end();
       ++cell, ++i)
    cell->set_user_index(i);

  UpdateFlags update_flags = integrator.update_flags();
  bool values_flag = update_flags & update_values;
  bool gradients_flag = update_flags & update_gradients;
  bool hessians_flag = update_flags & update_hessians;

  MeshWorker::IntegrationInfoBox<dim> info_box;
  info_box.cell_selector.add("solution", values_flag, gradients_flag, hessians_flag);
  info_box.boundary_selector.add("solution", values_flag, gradients_flag, hessians_flag);
  info_box.face_selector.add("solution", values_flag, gradients_flag, hessians_flag);
  const unsigned int degree = this->fe->tensor_degree();
  info_box.initialize_gauss_quadrature(degree + 2, degree + 2, degree + 2);
  if (integrator.cell_quadrature != 0)
  {
    info_box.cell_quadrature = *(integrator.cell_quadrature);
  }
  if (integrator.boundary_quadrature != 0)
  {
    info_box.boundary_quadrature = *(integrator.boundary_quadrature);
  }
  if (integrator.face_quadrature != 0)
  {
    info_box.face_quadrature = *(integrator.face_quadrature);
  }

  info_box.add_update_flags_all(update_flags);
  info_box.initialize(
    *this->fe, this->mapping, solution_data, Vector<double>(), &this->dof_handler.block_info());

  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());

  MeshWorker::Assembler::CellsAndFaces<double> assembler;
  AnyData out_data;
  BlockVector<double>* est = &errors;
  out_data.add(est, "cells");
  assembler.initialize(out_data, false);

  MeshWorker::LoopControl new_control;
  new_control.cells_first = false;
  MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(),
                                         dof_handler.end(),
                                         dof_info,
                                         info_box,
                                         integrator,
                                         assembler,
                                         new_control);
}

template <int dim>
void
AmandusApplicationSparse<dim>::error(BlockVector<double>& errors,
                                     const dealii::AnyData& solution_data,
                                     const ErrorIntegrator<dim>& integrator)
{
  errors.reinit(integrator.size());
  this->error(errors, solution_data, static_cast<const AmandusIntegrator<dim>&>(integrator));
}

template <int dim>
void
AmandusApplicationSparse<dim>::error(const dealii::AnyData& solution_data,
                                     const AmandusIntegrator<dim>& integrator,
                                     unsigned int num_errs)
{
  BlockVector<double> errors(num_errs);
  error(errors, solution_data, integrator);

  for (unsigned int i = 0; i < num_errs; ++i)
    deallog << "Error(" << i << "): " << errors.block(i).l2_norm() << std::endl;
}

template <int dim>
void
AmandusApplicationSparse<dim>::output_results(const unsigned int cycle, const AnyData* in) const
{
  DataOut<dim> data_out;
  unsigned int subdivisions = fe->tensor_degree();
  if (param != 0)
  {
    param->enter_subsection("Output");
    data_out.parse_parameters(*param);
    subdivisions = param->get_integer("Subdivisions");
    param->leave_subsection();
  }
  else
  {
    data_out.set_default_format(DataOutBase::vtu);
  }

  if (data_out.default_suffix() == std::string(""))
  {
    deallog << "No output cycle " << cycle << std::endl;
    return;
  }

  data_out.attach_dof_handler(dof_handler);
  if (in != 0)
  {
    for (unsigned int i = 0; i < in->size(); ++i)
      if (in->entry<Vector<double>*>(i)->size() == triangulation->n_active_cells())
        data_out.add_data_vector(*(in->entry<Vector<double>*>(i)), in->name(i));
      else
        data_out.add_data_vector(*(in->entry<Vector<double>*>(i)),
                                 in->name(i),
                                 DataOut_DoFData<DoFHandler<dim>, dim, dim>::type_dof_data,
                                 output_data_types);
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }
  data_out.build_patches(subdivisions);

  std::ostringstream filename;
  filename << "solution-" << std::setfill('0') << std::setw(3) << cycle
           << data_out.default_suffix();

  deallog << "Writing " << filename.str() << std::endl;

  std::ofstream output(filename.str().c_str());
  data_out.write(output);
}

template class AmandusApplicationSparse<2>;
template class AmandusApplicationSparse<3>;
