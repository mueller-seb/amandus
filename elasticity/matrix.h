// $Id$

#ifndef __elasticity_matrix_h
#define __elasticity_matrix_h

#include <deal.II/meshworker/integration_info.h>
#include <integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/elasticity.h>

using namespace dealii;
using namespace LocalIntegrators;


namespace Elasticity
{
/**
 * Integrator for Elasticity problems and heat equation.
 *
 * The distinction between stationary and instationary problems is
 * made by the variable AmandusIntegrator::timestep, which is
 * inherited from the base class. If this variable is zero, we solve a
 * stationary problem. If it is nonzero, we assemble for an implicit
 * scheme.
 *
 * @ingroup integrators
 */
template <int dim>
class Matrix : public AmandusIntegrator<dim>
{
public:
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
		    MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const;
};


template <int dim>
void Matrix<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  AssertDimension (dinfo.n_matrices(), 1);
  Elasticity::cell_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0));
}


template <int dim>
void Matrix<dim>::boundary(
  MeshWorker::DoFInfo<dim>& dinfo,
  typename MeshWorker::IntegrationInfo<dim>& info) const
{
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Elasticity::nitsche_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0),
			     Elasticity::compute_penalty(dinfo, dinfo, deg, deg));
}


template <int dim>
void Matrix<dim>::face(
  MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
  MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Elasticity::ip_matrix(dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix, 
			dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
			info1.fe_values(0), info2.fe_values(0),
			Elasticity::compute_penalty(dinfo1, dinfo2, deg, deg));
}
}


#endif