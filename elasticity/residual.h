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
    Residual(const Parameters& par);
    
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
};


//----------------------------------------------------------------------//

  template <int dim>
  Residual<dim>::Residual(const Parameters& par)
		  :
		  parameters(&par)
  {
//    this->use_boundary = false;
//    this->use_face = false;
    this->input_vector_names.push_back("Newton iterate");
  }
  
  template <int dim>
  void Residual<dim>::cell(
    DoFInfo<dim>& dinfo, 
    IntegrationInfo<dim>& info) const
  {
    Assert(info.values.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));
    Assert(info.gradients.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));
    
    dealii::LocalIntegrators::Elasticity::cell_residual(
      dinfo.vector(0).block(0), info.fe_values(0),
      dealii::make_slice(info.gradients[0], 0, dim), parameters->mu);
    dealii::LocalIntegrators::Divergence::grad_div_residual(
      dinfo.vector(0).block(0), info.fe_values(0),
      dealii::make_slice(info.gradients[0], 0, dim), parameters->lambda);
  }


template <int dim>
void Residual<dim>::boundary(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<std::vector<double> > null(dim, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));

  if (dinfo.face->boundary_indicator() == 0)
    std::fill(null[0].begin(), null[0].end(), -.1);
  if (dinfo.face->boundary_indicator() == 1)
    std::fill(null[0].begin(), null[0].end(), .1);
  
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  if (dinfo.face->boundary_indicator() == 0 || dinfo.face->boundary_indicator() == 1)
    dealii::LocalIntegrators::Elasticity::nitsche_residual(
      dinfo.vector(0).block(0), info.fe_values(0),
      dealii::make_slice(info.values[0], 0, dim),
      dealii::make_slice(info.gradients[0], 0, dim),
      null,
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      parameters->mu);
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
      parameters->mu);
  }
}


#endif
  
