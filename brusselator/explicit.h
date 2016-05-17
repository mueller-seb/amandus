/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __brusselator_explicit_h
#define __brusselator_explicit_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>
#include <amandus/integrator.h>
#include <amandus/brusselator/parameters.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

namespace Brusselator
{
/**
 * Integrate the residual for a Brusselator problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 *
 * @ingroup integrators
 */
  template <int dim>
  class ExplicitResidual : public AmandusIntegrator<dim>
  {
    public:
      ExplicitResidual(const Parameters& par);
    
      virtual void cell(DoFInfo<dim>& dinfo,
			IntegrationInfo<dim>& info) const;
      virtual void boundary(DoFInfo<dim>& dinfo,
			    IntegrationInfo<dim>& info) const;
      virtual void face(DoFInfo<dim>& dinfo1,
			DoFInfo<dim>& dinfo2,
			IntegrationInfo<dim>& info1,
			IntegrationInfo<dim>& info2) const;
    private:
      SmartPointer<const Parameters, class ImplicitResidual<dim> > parameters;
  };


//----------------------------------------------------------------------//

  template <int dim>
  ExplicitResidual<dim>::ExplicitResidual(const Parameters& par)
		  :
		  parameters(&par)
  {
    this->use_boundary = false;
    this->use_face = true;
  }


  template <int dim>
  void ExplicitResidual<dim>::cell(
    DoFInfo<dim>& dinfo, 
    IntegrationInfo<dim>& info) const
  {
    Assert (this->timestep != 0, ExcMessage("Only for transient problems"));
    Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
    Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  
    std::vector<double> rhs0 (info.fe_values(0).n_quadrature_points, 0.);
    std::vector<double> rhs1 (info.fe_values(0).n_quadrature_points, 0.);

    for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
      {
	const double u = info.values[0][0][k];
	const double v = info.values[0][1][k];
	rhs0[k] = u - this->timestep
		  * (-parameters->B - u*u*v + (parameters->A+1.)*u );
	rhs1[k] = v - this->timestep
		  * (-parameters->A*u + u*u*v );
      }
  
    L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs0);
    L2::L2(dinfo.vector(0).block(1), info.fe_values(0), rhs1);
    Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0),
    			   info.gradients[0][0], -parameters->alpha0*this->timestep);
    Laplace::cell_residual(dinfo.vector(0).block(1), info.fe_values(0),
    			   info.gradients[0][1], -parameters->alpha1*this->timestep);
  }


  template <int dim>
  void ExplicitResidual<dim>::boundary(
    DoFInfo<dim>&, 
    IntegrationInfo<dim>&) const
  {}


  template <int dim>
  void ExplicitResidual<dim>::face(
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
      -parameters->alpha0*this->timestep);
    Laplace::ip_residual(
      dinfo1.vector(0).block(1), dinfo2.vector(0).block(1),
      info1.fe_values(0), info2.fe_values(0),
      info1.values[0][1],
      info1.gradients[0][1],
      info2.values[0][1],
      info2.gradients[0][1],
      Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
      -parameters->alpha1*this->timestep);
  }
}


#endif
  
