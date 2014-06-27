/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __elasticity_residual_h
#define __elasticity_residual_h

#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/elasticity.h>
#include <deal.II/integrators/divergence.h>
#include <integrator.h>
#include <elasticity/parameters.h>
#include <elasticity/integrators.h>

using namespace dealii::MeshWorker;


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
    Residual(const Parameters& par, const dealii::Function<dim>& bdry);
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
    virtual void face(DoFInfo<dim>& dinfo1,
		      DoFInfo<dim>& dinfo2,
		      IntegrationInfo<dim>& info1,
		      IntegrationInfo<dim>& info2) const;
    private:
    dealii::SmartPointer<const Parameters, class Residual<dim> > parameters;
    dealii::SmartPointer<const dealii::Function<dim>, class Residual<dim> > boundary_values;
};


//----------------------------------------------------------------------//

  template <int dim>
  Residual<dim>::Residual(const Parameters& par, const dealii::Function<dim>& bdry)
		  :
		  parameters(&par),
		  boundary_values(&bdry)
  {
   this->use_boundary = false;
   this->use_face = false;
    this->input_vector_names.push_back("Newton iterate");
  }
  
  template <int dim>
  void Residual<dim>::cell(
    DoFInfo<dim>& dinfo, 
    IntegrationInfo<dim>& info) const
  {
    Assert(info.values.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));
    Assert(info.gradients.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));

    const double mu = parameters->mu;
    const double lambda = parameters->lambda;
    
    if (parameters->linear)
      {
	dealii::LocalIntegrators::Elasticity::cell_residual(
	  dinfo.vector(0).block(0), info.fe_values(0),
	  dealii::make_slice(info.gradients[0], 0, dim), 2.*mu);
	dealii::LocalIntegrators::Divergence::grad_div_residual(
	  dinfo.vector(0).block(0), info.fe_values(0),
	  dealii::make_slice(info.gradients[0], 0, dim), lambda);
      }
    else
      {
	Hooke_finite_strain_residual(dinfo.vector(0).block(0), info.fe_values(0),
				     dealii::make_slice(info.gradients[0], 0, dim),
				     lambda, mu);
      }
  }


template <int dim>
void Residual<dim>::boundary(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<std::vector<double> > bdry(dim, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));
  
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  if (dinfo.face->boundary_indicator() == 0 || dinfo.face->boundary_indicator() == 1)
    {
      boundary_values->vector_values(info.fe_values(0).get_quadrature_points(), bdry);
      dealii::LocalIntegrators::Elasticity::nitsche_residual(
	dinfo.vector(0).block(0), info.fe_values(0),
	dealii::make_slice(info.values[0], 0, dim),
	dealii::make_slice(info.gradients[0], 0, dim),
	bdry,
	dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
	2.*parameters->mu);
    }
}
  

  template <int dim>
  void Residual<dim>::face(
    DoFInfo<dim>& dinfo1, 
    DoFInfo<dim>& dinfo2,
    IntegrationInfo<dim>& info1, 
    IntegrationInfo<dim>& info2) const
  {
    const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
    dealii::LocalIntegrators::Elasticity::ip_residual(
      dinfo1.vector(0).block(0), dinfo2.vector(0).block(0),
      info1.fe_values(0), info2.fe_values(0),
      dealii::make_slice(info1.values[0], 0, dim),
      dealii::make_slice(info1.gradients[0], 0, dim),
      dealii::make_slice(info2.values[0], 0, dim),
      dealii::make_slice(info2.gradients[0], 0, dim),
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
      2.*parameters->mu);
  }
}


#endif
  










