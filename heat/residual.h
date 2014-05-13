/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __heat_residual_h
#define __heat_residual_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>

#ifndef M_PI
#define M_PI    3.14159265358979323846f
#endif

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

/**
 * Integrate the right hand side for a Laplace problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 */
template <int dim>
class HeatRHS : public LocalIntegrator<dim>
{
  public:
    HeatRHS();
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
    virtual void face(DoFInfo<dim>& dinfo1,
		      DoFInfo<dim>& dinfo2,
		      IntegrationInfo<dim>& info1,
		      IntegrationInfo<dim>& info2) const;
};


/**
 * Integrate the residual for a Laplace problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 */
template <int dim>
class HeatResidual : public LocalIntegrator<dim>
{
  public:
    HeatResidual();
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
    virtual void face(DoFInfo<dim>& dinfo1,
		      DoFInfo<dim>& dinfo2,
		      IntegrationInfo<dim>& info1,
		      IntegrationInfo<dim>& info2) const;
};


template <int dim>
class HeatError : public LocalIntegrator<dim>
{
  public:
    HeatError();
    
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
HeatRHS<dim>::HeatRHS()
{
  //this->use_boundary = false; // due to inhom boundary data
  //this->use_face = false;
}


template <int dim>
void HeatRHS<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
}


template <int dim>
void HeatRHS<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{
  //*********************
}


template <int dim>
void HeatRHS<dim>::face(
  DoFInfo<dim>&, 
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

//----------------------------------------------------------------------//

template <int dim>
HeatResidual<dim>::HeatResidual()
{
  this->use_boundary = true;
  this->use_face = true;
  this->input_vector_names.push_back("Newton iterate");
}


template <int dim>
void HeatResidual<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  
  /*  std::vector<double> rhs (info.fe_values(0).n_quadrature_points, 0.);

  std::vector<double> px(3);
  std::vector<double> py(3);
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
      const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
      solution_1d.value(x, px);
      solution_1d.value(y, py);
      
      rhs[k] = -px[2]*py[0]-px[0]*py[2];
    }
  */
  // L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs, -1.);
  Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0),
			 info.gradients[0][0]);
}


template <int dim>
void HeatResidual<dim>::boundary(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<double> boundary_data(info.fe_values(0).n_quadrature_points, 0.);
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
      const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
      
      boundary_data[k] = std::sin( 1*M_PI*x )* std::sin( 1*M_PI*y );
    }
  
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_residual(dinfo.vector(0).block(0), info.fe_values(0),
			    info.values[0][0],
			    info.gradients[0][0],
			    boundary_data,
			    Laplace::compute_penalty(dinfo, dinfo, deg, deg));
}


template <int dim>
void HeatResidual<dim>::face(
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
		  Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
}

//----------------------------------------------------------------------//

template <int dim>
HeatError<dim>::HeatError()
{
  this->use_boundary = false;
  this->use_face = false;
}


template <int dim>
void HeatError<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  Assert(dinfo.n_values() >= 2, ExcDimensionMismatch(dinfo.n_values(), 4));
  
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
      const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);

      const double dx = info.fe_values(0).JxW(k);
      
      Tensor<1,dim> Du = info.gradients[0][0][k];
      Du[0] -= 1*M_PI* std::cos( 1*M_PI*x )* std::sin( 1*M_PI*y );
      Du[1] -= std::sin( 1*M_PI*x )* 1*M_PI* std::cos( 1*M_PI*y );
      double u = info.values[0][0][k];
      u -= std::sin( 1*M_PI*x )* std::sin( 1*M_PI*y );

      // 0. L^2(u)
      dinfo.value(0) += (u*u) * dx;
      // 1. H^1(u)
      dinfo.value(1) += (Du*Du) * dx;
    }
  
}


template <int dim>
void HeatError<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}


template <int dim>
void HeatError<dim>::face(
  DoFInfo<dim>&, 
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

#endif
  
