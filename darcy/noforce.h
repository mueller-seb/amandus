/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __darcy_noforce_h
#define __darcy_noforce_h

#include <amandus/integrator.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

namespace DarcyIntegrators
{
/**
 * Integrate the residual for a Darcy problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 *
 * @ingroup integrators
 */
template <int dim>
class NoForceResidual : public AmandusIntegrator<dim>
{
public:
  DarcyNoForceResidual();
  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const;
};

//----------------------------------------------------------------------//

template <int dim>
inline DarcyNoForceResidual<dim>::DarcyNoForceResidual()
{
  this->input_vector_names.push_back("Newton iterate");
}

template <int dim>
inline void
DarcyNoForceResidual<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));

  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), make_slice(info.values[0], 0, dim));
  Divergence::gradient_residual(
    dinfo.vector(0).block(0), info.fe_values(0), info.values[0][dim], -1.);
  Divergence::cell_residual(
    dinfo.vector(0).block(1), info.fe_values(1), make_slice(info.gradients[0], 0, dim), 1.);
}

template <int dim>
inline void
DarcyNoForceResidual<dim>::boundary(DoFInfo<dim>&, IntegrationInfo<dim>&) const
{
}

template <int dim>
inline void
DarcyNoForceResidual<dim>::face(DoFInfo<dim>&, DoFInfo<dim>&, IntegrationInfo<dim>&,
                                IntegrationInfo<dim>&) const
{
}
}

#endif
