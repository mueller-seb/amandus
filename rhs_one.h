// $Id: rhs_one.h 1384 2014-01-10 14:04:00Z kanschat $

#ifndef __rhs_one_h
#define __rhs_one_h

#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>

#include <deal.II/fe/fe_values.h>

using namespace dealii;
using namespace LocalIntegrators;

/**
 * A local integrator for a right hand side equal to constant one in
 * the vector valued equation and zero in the scalar.
 */
template <int dim>
class RhsOne : public MeshWorker::LocalIntegrator<dim>
{
  public:
  RhsOne();
    virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
    virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
    virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
		     MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const;
};


template <int dim>
RhsOne<dim>::RhsOne()
{}


template <int dim>
void RhsOne<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values(0);
  const unsigned int n_dofs = fe.dofs_per_cell;
  Vector<double>& result = dinfo.vector(0).block(0);
  for (unsigned int k=0; k<fe.n_quadrature_points; ++k)
    for (unsigned int i=0; i<n_dofs; ++i)
      for (unsigned int d=0; d<dim; ++d)
	result(i) += fe.JxW(k) * fe.shape_value_component(i,k,d);
}


template <int dim>
void RhsOne<dim>::boundary(MeshWorker::DoFInfo<dim>& /*dinfo*/,
			   typename MeshWorker::IntegrationInfo<dim>& /*info*/) const
{}


template <int dim>
void RhsOne<dim>::face(MeshWorker::DoFInfo<dim>& /*dinfo1*/,
		       MeshWorker::DoFInfo<dim>& /*dinfo2*/,
		       MeshWorker::IntegrationInfo<dim>& /*info1*/,
		       MeshWorker::IntegrationInfo<dim>& /*info2*/) const
{}




#endif
