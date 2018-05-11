/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/relaxation_block.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/output.h>
#include <deal.II/meshworker/simple.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/flow_function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include <amandus/amandus.h>

using namespace dealii;

template <int dim, typename RELAXATION>
AmandusApplication<dim, RELAXATION>::AmandusApplication(Triangulation<dim>& triangulation,
                                                        const FiniteElement<dim>& fe)
  : AmandusApplicationSparse<dim>(triangulation, fe, false)
  , mg_transfer(mg_constraints)
{
}

template <int dim, typename RELAXATION>
void
AmandusApplication<dim, RELAXATION>::parse_parameters(dealii::ParameterHandler& new_param)
{
  AmandusApplicationSparse<dim>::parse_parameters(new_param);

  new_param.enter_subsection("Linear Solver");
  this->right_preconditioning = new_param.get_bool("Use Right Preconditioning");
  this->use_default_residual = new_param.get_bool("Use Default Residual");
  new_param.leave_subsection();

  new_param.enter_subsection("Multigrid");
  this->smoother_relaxation = new_param.get_double("Smoother Relaxation");
  this->log_smoother_statistics = new_param.get_bool("Log Smoother Statistics");
  new_param.leave_subsection();
}

template <int dim, typename RELAXATION>
void
AmandusApplication<dim, RELAXATION>::setup_system()
{
  mg_transfer.clear();
  AmandusApplicationSparse<dim>::setup_system();

  deallog << "DoFHandler levels: ";
  for (unsigned int l = 0; l < this->triangulation->n_levels(); ++l)
    deallog << ' ' << this->dof_handler.n_dofs(l);
  deallog << std::endl;

  mg_transfer.initialize_constraints(mg_constraints);
  mg_transfer.build_matrices(this->dof_handler);

  const unsigned int n_levels = this->triangulation->n_levels();
  mg_smoother.clear_elements();
  smoother_data.resize(0, n_levels - 1);
  mg_matrix.resize(0, n_levels - 1);
  mg_matrix.clear_elements();
  mg_matrix_up.resize(0, n_levels - 1);
  mg_matrix_up.clear_elements();
  mg_matrix_down.resize(0, n_levels - 1);
  mg_matrix_down.clear_elements();
  mg_matrix_flux_up.resize(0, n_levels - 1);
  mg_matrix_flux_up.clear_elements();
  mg_matrix_flux_down.resize(0, n_levels - 1);
  mg_matrix_flux_down.clear_elements();
  mg_sparsity.resize(0, n_levels - 1);
  mg_sparsity_fluxes.resize(0, n_levels - 1);

  for (unsigned int level = mg_sparsity.min_level(); level <= mg_sparsity.max_level(); ++level)
  {
    DynamicSparsityPattern c_sparsity(this->dof_handler.n_dofs(level));
    MGTools::make_flux_sparsity_pattern(this->dof_handler, c_sparsity, level);
    mg_sparsity[level].copy_from(c_sparsity);
    mg_matrix[level].reinit(mg_sparsity[level]);
    mg_matrix_up[level].reinit(mg_sparsity[level]);
    mg_matrix_down[level].reinit(mg_sparsity[level]);

    if (level > 0)
    {
      DynamicSparsityPattern dg_sparsity;
      dg_sparsity.reinit(this->dof_handler.n_dofs(level - 1), this->dof_handler.n_dofs(level));
      MGTools::make_flux_sparsity_pattern_edge(this->dof_handler, dg_sparsity, level);
      mg_sparsity_fluxes[level].copy_from(dg_sparsity);
      mg_matrix_flux_up[level].reinit(mg_sparsity_fluxes[level]);
      mg_matrix_flux_down[level].reinit(mg_sparsity_fluxes[level]);
    }
  }
}

template <int dim, typename RELAXATION>
void
AmandusApplication<dim, RELAXATION>::setup_constraints()
{
  AmandusApplicationSparse<dim>::setup_constraints();

  this->mg_constraints.clear();
  this->mg_constraints.initialize(this->dof_handler);
  const unsigned int n_comp = this->dof_handler.get_fe().n_components();
  for (std::map<dealii::types::boundary_id, dealii::ComponentMask>::const_iterator p =
         this->boundary_masks.begin();
       p != this->boundary_masks.end();
       ++p)
    if (p->second.n_selected_components(n_comp) != 0)
    {
      std::set<dealii::types::boundary_id> boundary_ids;
      boundary_ids.insert(p->first);
      this->mg_constraints.make_zero_boundary_constraints(
        this->dof_handler, boundary_ids, p->second);
    }
}

template <int dim, typename RELAXATION>
void
AmandusApplication<dim, RELAXATION>::assemble_mg_matrix(const dealii::AnyData& in,
                                                        const AmandusIntegrator<dim>& integrator)
{
  mg_matrix = 0.;

  std::vector<MGLevelObject<Vector<double>>> aux(integrator.input_vector_names.size());
  AnyData mg_in;

  MeshWorker::IntegrationInfoBox<dim> info_box;

  std::size_t in_idx = 0;
  typedef typename std::vector<std::string>::const_iterator it;
  for (it i = integrator.input_vector_names.begin(); i != integrator.input_vector_names.end();
       ++i, ++in_idx)
  {
    const unsigned int min_level = mg_sparsity.min_level();
    const unsigned int max_level = mg_sparsity.max_level();
    aux[in_idx].resize(min_level, max_level);
    mg_transfer.copy_to_mg(this->dof_handler, aux[in_idx], *(in.read_ptr<Vector<double>>(*i)));
    mg_in.add(&(aux[in_idx]), *i);
    info_box.cell_selector.add(*i, true, true, false);
    info_box.boundary_selector.add(*i, true, true, false);
    info_box.face_selector.add(*i, true, true, false);
  }

  UpdateFlags update_flags = integrator.update_flags();
  info_box.add_update_flags_all(update_flags);
  info_box.add_update_flags_boundary(integrator.update_flags_face());
  info_box.add_update_flags_face(integrator.update_flags_face());
  info_box.initialize(
    *this->fe, this->mapping, mg_in, Vector<double>(), &this->dof_handler.block_info());

  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());

  MeshWorker::Assembler::MGMatrixSimple<SparseMatrix<double>> assembler;
  assembler.initialize(mg_constraints);
  assembler.initialize(mg_matrix);
  assembler.initialize_interfaces(mg_matrix_up, mg_matrix_down);
  assembler.initialize_fluxes(mg_matrix_flux_up, mg_matrix_flux_down);

  MeshWorker::LoopControl new_control;
  new_control.cells_first = false;
  MeshWorker::integration_loop<dim, dim>(this->dof_handler.begin_mg(),
                                         this->dof_handler.end_mg(),
                                         dof_info,
                                         info_box,
                                         integrator,
                                         assembler,
                                         new_control);

  coarse_matrix.reinit(0, 0);
  coarse_matrix.copy_from(mg_matrix[mg_matrix.min_level()]);
  mg_coarse.initialize(coarse_matrix, 1.e-15);

  const unsigned int n_comp = this->dof_handler.get_fe().n_components();

  bool sort = false;
  bool interior_dofs_only = true;
  unsigned int smoothing_steps = 1;
  bool variable_smoothing_steps = false;
  BlockMask exclude_boundary_dofs(n_comp, true);
  if (this->param != 0)
  {
    this->param->enter_subsection("Multigrid");
    sort = this->param->get_bool("Sort");
    interior_dofs_only = this->param->get_bool("Interior smoothing");
    const std::string included_blocks_string =
      this->param->get("Include exterior smoothing on blocks");
    smoothing_steps = this->param->get_integer("Smoothing steps on leaves");
    variable_smoothing_steps = this->param->get_bool("Variable smoothing steps");
    if (interior_dofs_only == false)
    {
      std::vector<bool> exclude_block(n_comp, false);
      exclude_boundary_dofs = dealii::BlockMask(exclude_block);
    }
    else if (included_blocks_string != "")
    {
      // if not for all the blocks the boundary dofs on each cell are neglected,
      // assume first that all boundary dofs are taken into account
      std::vector<bool> exclude_block(n_comp, true);
      const std::vector<std::string> included_blocks_strings =
        dealii::Utilities::split_string_list(included_blocks_string, ',');
      const std::vector<int> included_blocks =
        dealii::Utilities::string_to_int(included_blocks_strings);
      for (unsigned int i = 0; i < included_blocks.size(); ++i)
      {
        AssertIndexRange(included_blocks[i], static_cast<int>(n_comp));
        exclude_block[included_blocks[i]] = false;
      }
      exclude_boundary_dofs = dealii::BlockMask(exclude_block);
    }
    this->param->leave_subsection();
  }

  for (unsigned int l = smoother_data.min_level() + 1; l <= smoother_data.max_level(); ++l)
  {
    std::vector<unsigned int> vertex_mapping;
    if (this->vertex_patches)
      vertex_mapping = DoFTools::make_vertex_patches(smoother_data[l].block_list,
                                                     this->dof_handler,
                                                     l,
                                                     exclude_boundary_dofs,
                                                     boundary_patches,
                                                     false,
                                                     false,
                                                     true);
    else
    {
      smoother_data[l].block_list.reinit(
        this->triangulation->n_cells(l), this->dof_handler.n_dofs(l), this->fe->dofs_per_cell);
      DoFTools::make_cell_patches(smoother_data[l].block_list, this->dof_handler, l);
    }
    // Here we sort the blocks made from vertex patches or from cell patches in a given direction
    // which is usefull in case of advection dominated problems.
    if (sort)
    {
      const unsigned int ndir = advection_directions.size();
      smoother_data[l].order.resize(ndir);
      for (unsigned int i = 0; i < ndir; ++i)
      {
        std::vector<unsigned int>& order_for_direction = smoother_data[l].order[i];
        order_for_direction.resize(smoother_data[l].block_list.n_rows());
        std::vector<std::pair<Point<dim>, types::global_dof_index>> aux(order_for_direction.size());
        if (this->vertex_patches)
        {
          for (unsigned int j = 0; j < vertex_mapping.size(); ++j)
          {
            aux[j].first = this->triangulation->get_vertices()[vertex_mapping[j]];
            aux[j].second = j;
          }
        }
        else
        {
          auto cell = this->triangulation->begin(l), endc = this->triangulation->end(l);
          unsigned int j = 0;
          for (; cell != endc; ++cell, ++j)
          {
            aux[j].first = cell->center();
            aux[j].second = j;
          }
        }
        DoFRenumbering::ComparePointwiseDownstream<dim> comp(advection_directions[i]);
        std::sort(aux.begin(), aux.end(), comp);
        order_for_direction.resize(aux.size());
        for (unsigned int j = 0; j < smoother_data[l].order[i].size(); ++j)
          order_for_direction[j] = aux[j].second;
      }
    }

    smoother_data[l].block_list.compress();
    smoother_data[l].relaxation = smoother_relaxation;
    smoother_data[l].inversion = PreconditionBlockBase<double>::svd;
    smoother_data[l].threshold = 1.e-12;
  }
  mg_smoother.initialize(mg_matrix, smoother_data);
  if (log_smoother_statistics)
    for (unsigned int l = smoother_data.min_level() + 1; l <= smoother_data.max_level(); ++l)
    {
      deallog << "Level " << l << ' ';
      mg_smoother[l].log_statistics();
    }

  mg_smoother.set_steps(smoothing_steps);
  mg_smoother.set_variable(variable_smoothing_steps);
}

template <int dim, typename RELAXATION>
void
AmandusApplication<dim, RELAXATION>::solve(Vector<double>& sol, const Vector<double>& rhs)
{
  SolverGMRES<Vector<double>>::AdditionalData solver_data(
    100, right_preconditioning, use_default_residual);
  SolverGMRES<Vector<double>> solver(this->control, solver_data);
  // SolverCG<Vector<double> >::AdditionalData solver_data(false, false, true, true);
  // SolverCG<Vector<double> > solver(control, solver_data);
  // SolverRichardson<Vector<double> > solver(control, .6);

  mg::Matrix<Vector<double>> mgmatrix(mg_matrix);
  mg::Matrix<Vector<double>> mgdown(mg_matrix_down);
  mg::Matrix<Vector<double>> mgup(mg_matrix_up);
  mg::Matrix<Vector<double>> mgfluxdown(mg_matrix_flux_down);
  mg::Matrix<Vector<double>> mgfluxup(mg_matrix_flux_up);

  Multigrid<Vector<double>> mg(
    mgmatrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother, mg_matrix.min_level());
  mg.set_edge_matrices(mgdown, mgup);
  mg.set_edge_flux_matrices(mgfluxdown, mgfluxup);

  PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>> preconditioner(
    this->dof_handler, mg, mg_transfer);
  solver.solve(this->matrix[0], sol, rhs, preconditioner);
  this->constraints().distribute(sol);
}

#ifdef DEAL_II_WITH_ARPACK
template <int dim, typename RELAXATION>
void
AmandusApplication<dim, RELAXATION>::arpack_solve(std::vector<std::complex<double>>& eigenvalues,
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
    AmandusApplicationSparse<dim>::setup_vector(arpack_vectors[i]);

  if (eigenvectors[0].l2_norm() != 0.)
    solver.set_initial_vector(eigenvectors[0]);

  mg::Matrix<Vector<double>> mgmatrix(mg_matrix);
  mg::Matrix<Vector<double>> mgdown(mg_matrix_down);
  mg::Matrix<Vector<double>> mgup(mg_matrix_up);
  mg::Matrix<Vector<double>> mgfluxdown(mg_matrix_flux_down);
  mg::Matrix<Vector<double>> mgfluxup(mg_matrix_flux_up);

  Multigrid<Vector<double>> mg(
    mgmatrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother, mg_matrix.min_level());
  mg.set_edge_matrices(mgdown, mgup);
  mg.set_edge_flux_matrices(mgfluxdown, mgfluxup);

  PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>> preconditioner(
    this->dof_handler, mg, mg_transfer);

  SolverGMRES<Vector<double>>::AdditionalData solver_data(40, true);
  SolverGMRES<Vector<double>> inner(this->control, solver_data);
  const auto A = linear_operator(this->matrix[0]); 
  const auto inv = inverse_operator(A, inner, preconditioner); 

  for (unsigned int i = 0; i < this->matrix[1].m(); ++i)
    if (this->constraints().is_constrained(i))
      this->matrix[1].diag_element(i) = 0.;

  solver.solve(
    this->matrix[0], this->matrix[1], inv, eigenvalues, arpack_vectors, eigenvalues.size());

  for (unsigned int i = 0; i < arpack_vectors.size(); ++i)
    this->constraints().distribute(arpack_vectors[i]);

  for (unsigned int i = 0; i < eigenvalues.size(); ++i)
  {
    eigenvectors[i] = arpack_vectors[i];
    if (std::fabs(eigenvalues[i].imag()) > 1.e-12)
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
template <int dim, typename RELAXATION>
void
AmandusApplication<dim, RELAXATION>::arpack_solve(
  std::vector<std::complex<double>>& /*eigenvalues*/, std::vector<Vector<double>>& /*eigenvectors*/)
{
  AssertThrow(false, ExcNeedArpack());
}
#endif

template class AmandusApplication<2, dealii::RelaxationBlockSSOR<dealii::SparseMatrix<double>>>;
template class AmandusApplication<3, dealii::RelaxationBlockSSOR<dealii::SparseMatrix<double>>>;
template class AmandusApplication<2, dealii::RelaxationBlockSOR<dealii::SparseMatrix<double>>>;
template class AmandusApplication<3, dealii::RelaxationBlockSOR<dealii::SparseMatrix<double>>>;
template class AmandusApplication<2, dealii::RelaxationBlockJacobi<dealii::SparseMatrix<double>>>;
template class AmandusApplication<3, dealii::RelaxationBlockJacobi<dealii::SparseMatrix<double>>>;
