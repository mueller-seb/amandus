/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __stokes_noforce_h
#define __stokes_noforce_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

/**
 * Integrate the residual for a Stokes problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 */
template <int dim>
class StokesNoForceResidual : public LocalIntegrator<dim>
{
  public:
    StokesNoForceResidual();
    
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
StokesNoForceResidual<dim>::StokesNoForceResidual()
{
  this->input_vector_names.push_back("Newton iterate");
}


template <int dim>
void StokesNoForceResidual<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  
  Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0),
			 make_slice(info.gradients[0], 0, dim));
  // This must be the weak gradient residual!
  Divergence::gradient_residual(dinfo.vector(0).block(0), info.fe_values(0),
  				info.values[0][dim], -1.);
  Divergence::cell_residual(dinfo.vector(0).block(1), info.fe_values(1),
  			    make_slice(info.gradients[0], 0, dim), 1.);
}


template <int dim>
void StokesNoForceResidual<dim>::boundary(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<std::vector<double> > null(dim, std::vector<double> (info.fe_values(0).n_quadrature_points, 0.));
  
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_residual(dinfo.vector(0).block(0), info.fe_values(0),
			    make_slice(info.values[0], 0, dim),
			    make_slice(info.gradients[0], 0, dim),
			    null,
			    Laplace::compute_penalty(dinfo, dinfo, deg, deg));
}


template <int dim>
void StokesNoForceResidual<dim>::face(
  DoFInfo<dim>& dinfo1, 
  DoFInfo<dim>& dinfo2,
  IntegrationInfo<dim>& info1, 
  IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Laplace::ip_residual(dinfo1.vector(0).block(0), dinfo2.vector(0).block(0),
		  info1.fe_values(0), info2.fe_values(0),
		  make_slice(info1.values[0], 0, dim),
		  make_slice(info1.gradients[0], 0, dim),
		  make_slice(info2.values[0], 0, dim),
		  make_slice(info2.gradients[0], 0, dim),
		  Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
}

#endif
  