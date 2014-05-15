/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __laplace_noforce_h
#define __laplace_noforce_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>
#include <integrator.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

/**
 * Integrate the "residual" of the heat equation which can be found on
 * the right hand side of the ThetaTimestepping operator.
 *
 * @ingroup integrators
 */
template <int dim>
class LaplaceThetaResidual : public AmandusIntegrator<dim>
{
  public:
    LaplaceThetaResidual();
    
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
LaplaceThetaResidual<dim>::LaplaceThetaResidual()
{
  this->input_vector_names.push_back("Previous iterate");
}


template <int dim>
void LaplaceThetaResidual<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  
  if (this->timestepping == 0.)
    Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0),
			   info.gradients[0][0]);
  else
    {
      Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0),
			     info.gradients[0][0], this->timestep);
      L2::L2(dinfo.vector(0).block(0), info.fe_values(0), info.values[0][0]);
    }
}


template <int dim>
void LaplaceThetaResidual<dim>::boundary(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<double> null(info.fe_values(0).n_quadrature_points, 0.);
  
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();

  Laplace::nitsche_residual(dinfo.vector(0).block(0), info.fe_values(0),
			    info.values[0][0],
			    info.gradients[0][0],
			    null,
			    Laplace::compute_penalty(dinfo, dinfo, deg, deg),
			    (this->timestep==0. ? 1. this->timestep));
}


template <int dim>
void LaplaceThetaResidual<dim>::face(
  DoFInfo<dim>& dinfo1, 
  DoFInfo<dim>& dinfo2,
  IntegrationInfo<dim>& info1, 
  IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Laplace::ip_residual(dinfo1.vector(0).block(0), dinfo2.vector(0).block(0),
		       info1.fe_values(0), info2.fe_values(0),
		       info1.values[0][0],
		       info1.gradients[0][0],
		       info2.values[0][0],
		       info2.gradients[0][0],
		       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
		       (this->timestep==0. ? 1. this->timestep));
}


#endif
  
