/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/iterative_inverse.h>
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
  , mg_transfer(this->constraint_matrix, mg_constraints)
{
}


template <int dim, typename RELAXATION>
void
AmandusApplication<dim, RELAXATION>::parse_parameters(dealii::ParameterHandler& param)
{
  AmandusApplicationSparse<dim>::parse_parameters(param);

  param.enter_subsection("Linear Solver");
  this->right_preconditioning = param.get_bool("Use Right Preconditioning");
  this->use_default_residual = param.get_bool("Use Default Residual");
  param.leave_subsection();

  param.enter_subsection("Multigrid");
  this->smoother_relaxation = param.get_double("Smoother Relaxation");
  this->log_smoother_statistics = param.get_bool("Log Smoother Statistics");
  param.leave_subsection();
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

  mg_transfer.initialize_constraints(this->constraint_matrix, mg_constraints);
  mg_transfer.build_matrices(this->dof_handler);

  const unsigned int n_levels = this->triangulation->n_levels();
  mg_smoother.clear();
  smoother_data.resize(0, n_levels - 1);
  mg_matrix.resize(0, n_levels - 1);
  mg_matrix.clear();
  mg_matrix_up.resize(0, n_levels - 1);
  mg_matrix_up.clear();
  mg_matrix_down.resize(0, n_levels - 1);
  mg_matrix_down.clear();
  mg_matrix_flux_up.resize(0, n_levels - 1);
  mg_matrix_flux_up.clear();
  mg_matrix_flux_down.resize(0, n_levels - 1);
  mg_matrix_flux_down.clear();
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
  info_box.initialize(
    *this->fe, this->mapping, mg_in, Vector<double>(), &this->dof_handler.block_info());

  MeshWorker::DoFInfo<dim> dof_info(this->dof_handler.block_info());

  MeshWorker::Assembler::MGMatrixSimple<SparseMatrix<double>> assembler;
  assembler.initialize(mg_constraints);
  assembler.initialize(mg_matrix);
  assembler.initialize_interfaces(mg_matrix_up, mg_matrix_down);
  assembler.initialize_fluxes(mg_matrix_flux_up, mg_matrix_flux_down);

  MeshWorker::LoopControl control;
  control.cells_first = false;
  MeshWorker::integration_loop<dim, dim>(this->dof_handler.begin_mg(),
                                         this->dof_handler.end_mg(),
                                         dof_info,
                                         info_box,
                                         integrator,
                                         assembler,
                                         control);

  coarse_matrix.reinit(0, 0);
  coarse_matrix.copy_from(mg_matrix[mg_matrix.min_level()]);
  mg_coarse.initialize(coarse_matrix, 1.e-15);

  bool sort = false;
  bool interior_dofs_only = true;
  unsigned int smoothing_steps = 1;
  bool variable_smoothing_steps = false;
  if (this->param != 0)
  {
    this->param->enter_subsection("Multigrid");
    sort = this->param->get_bool("Sort");
    interior_dofs_only = this->param->get_bool("Interior smoothing");
    smoothing_steps = this->param->get_integer("Smoothing steps on leaves");
    variable_smoothing_steps = this->param->get_bool("Variable smoothing steps");
    this->param->leave_subsection();
  }

  for (unsigned int l = smoother_data.min_level() + 1; l <= smoother_data.max_level(); ++l)
  {
    std::vector<unsigned int> vertex_mapping;
    if (this->vertex_patches)
      vertex_mapping = DoFTools::make_vertex_patches(smoother_data[l].block_list,
                                                     this->dof_handler,
                                                     l,
                                                     interior_dofs_only,
                                                     false,
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
        smoother_data[l].order[i].resize(smoother_data[l].block_list.n_rows());
        std::vector<std::pair<Point<dim>, types::global_dof_index>> aux(
          smoother_data[l].order[i].size());
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
        smoother_data[l].order[i].resize(aux.size());
        for (unsigned int j = 0; j < smoother_data[l].order[i].size(); ++j)
          smoother_data[l].order[i][j] = aux[j].second;
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
    this->dof_handler, mgmatrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
  mg.set_edge_matrices(mgdown, mgup);
  mg.set_edge_flux_matrices(mgfluxdown, mgfluxup);
  mg.set_minlevel(mg_matrix.min_level());
  mg.set_debug(0);

  PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>> preconditioner(
    this->dof_handler, mg, mg_transfer);
  try
  {
    solver.solve(this->matrix[0], sol, rhs, preconditioner);
  }
  catch (...)
  {
  }
  this->constraints().distribute(sol);
}

#ifdef DEAL_II_WITH_ARPACK
template <int dim, typename RELAXATION>
void
AmandusApplication<dim, RELAXATION>::arpack_solve(std::vector<std::complex<double>>& eigenvalues,
                                                  std::vector<Vector<double>>& eigenvectors)
{
  AssertDimension(2 * eigenvalues.size(), eigenvectors.size());
  ArpackSolver::AdditionalData arpack_data(std::max((unsigned long)20, 2 * eigenvectors.size() + 1),
                                           ArpackSolver::largest_magnitude);
  ArpackSolver solver(this->control, arpack_data);

  SolverGMRES<Vector<double>>::AdditionalData solver_data(40, true);
  mg::Matrix<Vector<double>> mgmatrix(mg_matrix);
  mg::Matrix<Vector<double>> mgdown(mg_matrix_down);
  mg::Matrix<Vector<double>> mgup(mg_matrix_up);
  mg::Matrix<Vector<double>> mgfluxdown(mg_matrix_flux_down);
  mg::Matrix<Vector<double>> mgfluxup(mg_matrix_flux_up);

  Multigrid<Vector<double>> mg(
    this->dof_handler, mgmatrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
  mg.set_edge_matrices(mgdown, mgup);
  mg.set_edge_flux_matrices(mgfluxdown, mgfluxup);
  mg.set_minlevel(mg_matrix.min_level());

  PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>> preconditioner(
    this->dof_handler, mg, mg_transfer);

  PreconditionIdentity identity;
  IterativeInverse<Vector<double>> inv;
  inv.initialize(this->matrix[0], preconditioner);
  // inv.initialize(this->matrix[0], identity);
  inv.solver.set_control(this->control);
  inv.solver.set_data(solver_data);
  inv.solver.select("gmres");

  for (unsigned int i = 0; i < this->matrix[1].m(); ++i)
    if (this->constraints().is_constrained(i))
      this->matrix[1].diag_element(i) = 0.;

  solver.solve(
    this->matrix[0], this->matrix[1], inv, eigenvalues, eigenvectors, eigenvalues.size());

  for (unsigned int i = 0; i < eigenvectors.size(); ++i)
    this->constraints().distribute(eigenvectors[i]);
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
