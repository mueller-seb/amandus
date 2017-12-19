/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __rhs_one_h
#define __rhs_one_h

#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>

#include <deal.II/fe/fe_values.h>

using namespace dealii;
using namespace LocalIntegrators;

/**
 * \brief Very simple local integrator for constant right hand side.
 *
 * A local integrator for a right hand side equal to constant one in
 * the scalar case and equal to the vector with constant one in every
 * entry, respectively.
 */
template <int dim>
class RhsOne : public AmandusIntegrator<dim>
{
  const unsigned int components;
public:
  /// Constructor, with argument system dimension.
  RhsOne(const unsigned int components=dim)
    :components(components)
  {}
  
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo,
                        MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                    MeshWorker::IntegrationInfo<dim>& info1,
                    MeshWorker::IntegrationInfo<dim>& info2) const;
};

template <int dim>
void
RhsOne<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values(0);
  const unsigned int n_dofs = fe.dofs_per_cell;
  Vector<double>& result = dinfo.vector(0).block(0);
  for (unsigned int k = 0; k < fe.n_quadrature_points; ++k)
    for (unsigned int i = 0; i < n_dofs; ++i)
      for (unsigned int d = 0; d < components; ++d)
        result(i) += fe.JxW(k) * fe.shape_value_component(i, k, d);
}

template <int dim>
void
RhsOne<dim>::boundary(MeshWorker::DoFInfo<dim>& /*dinfo*/,
                      typename MeshWorker::IntegrationInfo<dim>& /*info*/) const
{
}

template <int dim>
void
RhsOne<dim>::face(MeshWorker::DoFInfo<dim>& /*dinfo1*/, MeshWorker::DoFInfo<dim>& /*dinfo2*/,
                  MeshWorker::IntegrationInfo<dim>& /*info1*/,
                  MeshWorker::IntegrationInfo<dim>& /*info2*/) const
{
}

#endif
