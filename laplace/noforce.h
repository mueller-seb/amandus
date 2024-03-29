/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __laplace_noforce_h
#define __laplace_noforce_h

#include <amandus/integrator.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

namespace LaplaceIntegrators
{
/**
 * Integrate the right hand side for a Laplace problem with zero right hand side.
 *
 * @ingroup integrators
 */
template <int dim>
class NoForceRHS : public AmandusIntegrator<dim>
{
public:
  NoForceRHS();

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;
};

/**
 * Integrate the residual for a Laplace problem with zero right hand side.
 *
 * @ingroup integrators
 */
template <int dim>
class NoForceResidual : public AmandusIntegrator<dim>
{
public:
  NoForceResidual();

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;
};

//----------------------------------------------------------------------//

template <int dim>
NoForceRHS<dim>::NoForceRHS()
{
  this->use_boundary = false;
  this->use_face = false;
}

template <int dim>
void
NoForceRHS<dim>::cell(DoFInfo<dim>&, IntegrationInfo<dim>&) const
{
}

template <int dim>
void
NoForceRHS<dim>::boundary(DoFInfo<dim>&, IntegrationInfo<dim>&) const
{
}

template <int dim>
void
NoForceRHS<dim>::face(DoFInfo<dim>&, DoFInfo<dim>&, IntegrationInfo<dim>&,
                      IntegrationInfo<dim>&) const
{
}

//----------------------------------------------------------------------//

template <int dim>
NoForceResidual<dim>::NoForceResidual()
{
  this->input_vector_names.push_back("Newton iterate");
}

template <int dim>
void
NoForceResidual<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));

  std::vector<std::vector<double>> rhs(1,
                                       std::vector<double>(info.fe_values(0).n_quadrature_points));
  Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0), info.gradients[0][0]);
}

template <int dim>
void
NoForceResidual<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  std::vector<double> null(info.fe_values(0).n_quadrature_points, 0.);

  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_residual(dinfo.vector(0).block(0),
                            info.fe_values(0),
                            info.values[0][0],
                            info.gradients[0][0],
                            null,
                            Laplace::compute_penalty(dinfo, dinfo, deg, deg));
}

template <int dim>
void
NoForceResidual<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                           IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Laplace::ip_residual(dinfo1.vector(0).block(0),
                       dinfo2.vector(0).block(0),
                       info1.fe_values(0),
                       info2.fe_values(0),
                       info1.values[0][0],
                       info1.gradients[0][0],
                       info2.values[0][0],
                       info2.gradients[0][0],
                       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
}
}

#endif
