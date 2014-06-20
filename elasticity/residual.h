/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __elasticity_residual_h
#define __elasticity_residual_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/elasticity.h>
#include <deal.II/integrators/divergence.h>
#include <integrator.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;


namespace Elasticity
{
/**
 * Integrate the residual for a Laplace problem with zero right hand side.
 *
 * @ingroup integrators
 */
template <int dim>
class Residual : public AmandusIntegrator<dim>
{
  public:
    Residual();
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
    virtual void face(DoFInfo<dim>& dinfo1,
		      DoFInfo<dim>& dinfo2,
		      IntegrationInfo<dim>& info1,
		      IntegrationInfo<dim>& info2) const;
};


//----------------------------------------------------------------------//

template <int dim>
Residual<dim>::Residual()
{
  this->input_vector_names.push_back("Newton iterate");
}


template <int dim>
void Residual<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  
  std::vector<std::vector<double> > rhs (1,
					 std::vector<double>(info.fe_values(0).n_quadrature_points));
  Elasticity::cell_residual(dinfo.vector(0).block(0), info.fe_values(0),
			    make_slice(info.gradients[0][0], 0, dim));
}


template <int dim>
void Residual<dim>::boundary(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<double> null(info.fe_values(0).n_quadrature_points, 0.);
  
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Elasticity::nitsche_residual(dinfo.vector(0).block(0), info.fe_values(0),
			    make_slice(info.values[0][0], 0, dim),
			    make_slice(info.gradients[0][0], 0, dim),
			    null,
			    Elasticity::compute_penalty(dinfo, dinfo, deg, deg));
}


template <int dim>
void Residual<dim>::face(
  DoFInfo<dim>& dinfo1, 
  DoFInfo<dim>& dinfo2,
  IntegrationInfo<dim>& info1, 
  IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Elasticity::ip_residual(
    dinfo1.vector(0).block(0), dinfo2.vector(0).block(0),
    info1.fe_values(0), info2.fe_values(0),
    make_slice(info1.values[0][0], 0, dim),
    make_slice(info1.gradients[0][0], 0, dim),
    make_slice(info2.values[0][0], 0, dim),
    make_slice(info2.gradients[0][0], 0, dim),
    Elasticity::compute_penalty(dinfo1, dinfo2, deg, deg));
}
}


#endif
  