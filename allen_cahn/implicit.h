/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2014
 *
 **********************************************************************/

#ifndef __allen_cahn_implicit_h
#define __allen_cahn_implicit_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>
#include <integrator.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

namespace AllenCahn
{
/**
 * Integrate the residual for a AllenCahn problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 *
 * @ingroup integrators
 */
  template <int dim>
  class ImplicitResidual : public AmandusIntegrator<dim>
  {
    public:
      ImplicitResidual(double difusion);
    
      virtual void cell(DoFInfo<dim>& dinfo,
			IntegrationInfo<dim>& info) const;
      virtual void boundary(DoFInfo<dim>& dinfo,
			    IntegrationInfo<dim>& info) const;
      virtual void face(DoFInfo<dim>& dinfo1,
			DoFInfo<dim>& dinfo2,
			IntegrationInfo<dim>& info1,
			IntegrationInfo<dim>& info2) const;
    private:
      double D;
  };


//----------------------------------------------------------------------//

  template <int dim>
  ImplicitResidual<dim>::ImplicitResidual(double diffusion)
		  :
		  D(diffusion)
  {
    this->use_boundary = false;
    this->use_face = true;
  }


  template <int dim>
  void ImplicitResidual<dim>::cell(
    DoFInfo<dim>& dinfo, 
    IntegrationInfo<dim>& info) const
  {
    Assert (this->timestep != 0, ExcMessage("Only for transient problems"));
    Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
    Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  
    std::vector<double> rhs (info.fe_values(0).n_quadrature_points, 0.);

    for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
      {
	const double u = info.values[0][0][k];
	rhs[k] = u + this->timestep * u*(u*u-1.);
      }
  
    L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
    
    Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0),
			   info.gradients[0][0], D*this->timestep);
  }


  template <int dim>
  void ImplicitResidual<dim>::boundary(
    DoFInfo<dim>&, 
    IntegrationInfo<dim>&) const
  {}


  template <int dim>
  void ImplicitResidual<dim>::face(
    DoFInfo<dim>& dinfo1, 
    DoFInfo<dim>& dinfo2,
    IntegrationInfo<dim>& info1, 
    IntegrationInfo<dim>& info2) const
  {
    const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
    Laplace::ip_residual(
      dinfo1.vector(0).block(0), dinfo2.vector(0).block(0),
      info1.fe_values(0), info2.fe_values(0),
      info1.values[0][0],
      info1.gradients[0][0],
      info2.values[0][0],
      info2.gradients[0][0],
      Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
      D*this->timestep);
  }
}


#endif
  
