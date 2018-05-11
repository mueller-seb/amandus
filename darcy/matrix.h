/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef _amandus_matrix_darcy_h
#define _amandus_matrix_darcy_h

#include <amandus/integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/meshworker/integration_info.h>

using namespace dealii;
using namespace LocalIntegrators;

/**
 * Integrators for Darcy equation
 */
namespace DarcyIntegrators
{
template <int dim>
class Matrix : public AmandusIntegrator<dim>
{
public:
  Matrix();
  virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                    dealii::MeshWorker::IntegrationInfo<dim>& info) const override;
  virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const override;
  virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
                    dealii::MeshWorker::DoFInfo<dim>& dinfo2,
                    dealii::MeshWorker::IntegrationInfo<dim>& info1,
                    dealii::MeshWorker::IntegrationInfo<dim>& info2) const override;

  std::vector<double> resistance;
};

template <int dim>
Matrix<dim>::Matrix()
  : resistance(1, 1.)
{
  this->use_boundary = false;
  this->use_face = false;
}

template <int dim>
void
Matrix<dim>::cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                  dealii::MeshWorker::IntegrationInfo<dim>& info) const
{
  AssertDimension(dinfo.n_matrices(), 4);
  const unsigned int id = dinfo.cell->material_id();
  AssertIndexRange(id, resistance.size());
  const double R = resistance[id];

  L2::mass_matrix(dinfo.matrix(0, false).matrix, info.fe_values(0), R);
  Divergence::cell_matrix(dinfo.matrix(2, false).matrix, info.fe_values(0), info.fe_values(1));
  dinfo.matrix(1, false).matrix.copy_transposed(dinfo.matrix(2, false).matrix);
}

template <int dim>
void
Matrix<dim>::boundary(dealii::MeshWorker::DoFInfo<dim>& /*dinfo*/,
                      typename dealii::MeshWorker::IntegrationInfo<dim>& /*info*/) const
{
}

template <int dim>
void
Matrix<dim>::face(dealii::MeshWorker::DoFInfo<dim>& /*dinfo1*/,
                  dealii::MeshWorker::DoFInfo<dim>& /*dinfo2*/,
                  dealii::MeshWorker::IntegrationInfo<dim>& /*info1*/,
                  dealii::MeshWorker::IntegrationInfo<dim>& /*info2*/) const
{
}
}

#endif
