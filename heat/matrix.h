/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __matrix_heat_h
#define __matrix_heat_h

#include <amandus/integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/integration_info.h>

using namespace dealii;
using namespace LocalIntegrators;

/**
 * Integrator for Laplace problems and heat equation.
 *
 * The distinction between stationary and instationary problems is
 * made by the variable AmandusIntegrator::timestep, which is
 * inherited from the base class. If this variable is zero, we solve a
 * stationary problem. If it is nonzero, we assemble for an implicit
 * scheme.
 *
 * @ingroup integrators
 */
namespace HeatIntegrators
{
/**
 * \brief Integrator for the matrix of the Laplace operator.
 */
template <int dim>
class Matrix : public AmandusIntegrator<dim>
{
public:
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo,
                        MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                    MeshWorker::IntegrationInfo<dim>& info1,
                    MeshWorker::IntegrationInfo<dim>& info2) const;
};

template <int dim>
void
Matrix<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  AssertDimension(dinfo.n_matrices(), 1);
  Laplace::cell_matrix(dinfo.matrix(0, false).matrix, info.fe_values(0));
}

template <int dim>
void
Matrix<dim>::boundary(MeshWorker::DoFInfo<dim>& dinfo,
                      typename MeshWorker::IntegrationInfo<dim>& info) const
{
  if (info.fe_values(0).get_fe().conforms(FiniteElementData<dim>::H1))
    return;

  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_matrix(dinfo.matrix(0, false).matrix,
                          info.fe_values(0),
                          Laplace::compute_penalty(dinfo, dinfo, deg, deg));
}

template <int dim>
void
Matrix<dim>::face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                  MeshWorker::IntegrationInfo<dim>& info1,
                  MeshWorker::IntegrationInfo<dim>& info2) const
{
  if (info1.fe_values(0).get_fe().conforms(FiniteElementData<dim>::H1))
    return;

  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Laplace::ip_matrix(dinfo1.matrix(0, false).matrix,
                     dinfo1.matrix(0, true).matrix,
                     dinfo2.matrix(0, true).matrix,
                     dinfo2.matrix(0, false).matrix,
                     info1.fe_values(0),
                     info2.fe_values(0),
                     Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
}
}

#endif