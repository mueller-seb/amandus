/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __darcy_polynomial_h
#define __darcy_polynomial_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

/**
 * Integrate the right hand side for a Darcy problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 */
template <int dim>
class DarcyPolynomialRHS : public LocalIntegrator<dim>
{
  public:
    DarcyPolynomialRHS(const Polynomials::Polynomial<double> curl_potential_1d,
			const Polynomials::Polynomial<double> grad_potential_1d);
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
    virtual void face(DoFInfo<dim>& dinfo1,
		      DoFInfo<dim>& dinfo2,
		      IntegrationInfo<dim>& info1,
		      IntegrationInfo<dim>& info2) const;
  private:
    Polynomials::Polynomial<double> curl_potential_1d;
    Polynomials::Polynomial<double> grad_potential_1d;
};


/**
 * Integrate the residual for a Darcy problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 */
template <int dim>
class DarcyPolynomialResidual : public LocalIntegrator<dim>
{
  public:
    DarcyPolynomialResidual(const Polynomials::Polynomial<double> curl_potential_1d,
			const Polynomials::Polynomial<double> grad_potential_1d);
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
    virtual void face(DoFInfo<dim>& dinfo1,
		      DoFInfo<dim>& dinfo2,
		      IntegrationInfo<dim>& info1,
		      IntegrationInfo<dim>& info2) const;
  private:
    Polynomials::Polynomial<double> curl_potential_1d;
    Polynomials::Polynomial<double> grad_potential_1d;
};


template <int dim>
class DarcyPolynomialError : public LocalIntegrator<dim>
{
  public:
    DarcyPolynomialError(const Polynomials::Polynomial<double> curl_potential_1d,
			const Polynomials::Polynomial<double> grad_potential_1d);
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
    virtual void face(DoFInfo<dim>& dinfo1,
		      DoFInfo<dim>& dinfo2,
		      IntegrationInfo<dim>& info1,
		      IntegrationInfo<dim>& info2) const;
  private:
    Polynomials::Polynomial<double> curl_potential_1d;
    Polynomials::Polynomial<double> grad_potential_1d;
};

//----------------------------------------------------------------------//

template <int dim>
DarcyPolynomialRHS<dim>::DarcyPolynomialRHS(
  const Polynomials::Polynomial<double> curl_potential_1d,
  const Polynomials::Polynomial<double> grad_potential_1d)
		:
		curl_potential_1d(curl_potential_1d),
		grad_potential_1d(grad_potential_1d)
{
  this->use_boundary = false;
  this->use_face = false;
}


template <int dim>
void DarcyPolynomialRHS<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<std::vector<double> > rhs (dim,
					 std::vector<double>(info.fe_values(0).n_quadrature_points));

  std::vector<double> px(4);
  std::vector<double> py(4);
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
      const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
      curl_potential_1d.value(x, px);
      curl_potential_1d.value(y, py);
      
      rhs[0][k] = -px[2]*py[1]-px[0]*py[3];
      rhs[1][k] =  px[3]*py[0]+px[1]*py[2];

				       // Add a gradient part to the
				       // right hand side to test for
				       // pressure
      grad_potential_1d.value(x, px);
      grad_potential_1d.value(y, py);
      rhs[0][k] += px[1]*py[0];
      rhs[1][k] += px[0]*py[1];
    }
  
  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
}


template <int dim>
void DarcyPolynomialRHS<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}


template <int dim>
void DarcyPolynomialRHS<dim>::face(
  DoFInfo<dim>&, 
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

//----------------------------------------------------------------------//

template <int dim>
DarcyPolynomialResidual<dim>::DarcyPolynomialResidual(
  const Polynomials::Polynomial<double> curl_potential_1d,
  const Polynomials::Polynomial<double> grad_potential_1d)
		:
		curl_potential_1d(curl_potential_1d),
		grad_potential_1d(grad_potential_1d)
{
  this->use_boundary = true;
  this->use_face = true;
  this->input_vector_names.push_back("Newton iterate");
}


template <int dim>
void DarcyPolynomialResidual<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  
  std::vector<std::vector<double> > rhs (dim,
					 std::vector<double>(info.fe_values(0).n_quadrature_points));

  std::vector<double> px(4);
  std::vector<double> py(4);
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
      const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
      curl_potential_1d.value(x, px);
      curl_potential_1d.value(y, py);
      
      rhs[0][k] = -px[2]*py[1]-px[0]*py[3];
      rhs[1][k] =  px[3]*py[0]+px[1]*py[2];

				       // Add a gradient part to the
				       // right hand side to test for
				       // pressure
      grad_potential_1d.value(x, px);
      grad_potential_1d.value(y, py);
      rhs[0][k] += px[1]*py[0];
      rhs[1][k] += px[0]*py[1];
    }
  
  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs, -1.);
  L2::L2(dinfo.vector(0).block(0), info.fe_values(0),
			 make_slice(info.values[0], 0, dim));
  Divergence::gradient_residual(dinfo.vector(0).block(0), info.fe_values(0),
  				info.values[0][dim], -1.);
  Divergence::cell_residual(dinfo.vector(0).block(1), info.fe_values(1),
  			    make_slice(info.gradients[0], 0, dim), 1.);
}


template <int dim>
void DarcyPolynomialResidual<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}


template <int dim>
void DarcyPolynomialResidual<dim>::face(
  DoFInfo<dim>&, 
  DoFInfo<dim>&,
  IntegrationInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

//----------------------------------------------------------------------//

template <int dim>
DarcyPolynomialError<dim>::DarcyPolynomialError(
  const Polynomials::Polynomial<double> curl_potential_1d,
  const Polynomials::Polynomial<double> grad_potential_1d)
		:
		curl_potential_1d(curl_potential_1d),
		grad_potential_1d(grad_potential_1d)
{
  this->use_boundary = false;
  this->use_face = false;
}


template <int dim>
void DarcyPolynomialError<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
//  Assert(dinfo.n_values() >= 4, ExcDimensionMismatch(dinfo.n_values(), 4));
  
  std::vector<double> px(3);
  std::vector<double> py(3);
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
      const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
      curl_potential_1d.value(x, px);
      curl_potential_1d.value(y, py);
      const double dx = info.fe_values(0).JxW(k);
      
      Tensor<1,dim> Du0 = info.gradients[0][0][k];
      Du0[0] -= px[1]*py[1];
      Du0[1] -= px[0]*py[2];
      Tensor<1,dim> Du1 = info.gradients[0][1][k];
      Du1[0] += px[2]*py[0];
      Du1[1] += px[1]*py[1];
      double u0 = info.values[0][0][k];
      u0 -= px[0]*py[1];
      double u1 = info.values[0][1][k];
      u1 += px[1]*py[0];

      grad_potential_1d.value(x, px);
      grad_potential_1d.value(y, py);
      double p = info.values[0][dim][k];
      p += px[0]*py[0];
      Tensor<1,dim> Dp = info.gradients[0][dim][k];
      Dp[0] += px[1]*py[0];
      Dp[1] += px[0]*py[1];
      unsigned int i=0;
      // 0. L^2(u)
      dinfo.value(i++) += (u0*u0+u1*u1) * dx;
      // 1. H^1(u)
      dinfo.value(i++) += ((Du0*Du0)+(Du1*Du1)) * dx;
      // 2. div u
      dinfo.value(i++) =
	Divergence::norm(info.fe_values(0), make_slice(info.gradients[0], 0, dim));
      // 3. L^2(p) up to mean value
      dinfo.value(i++) += p*p * dx;
      // 4. H^1(p)
      dinfo.value(i++) += Dp*Dp * dx;
    }
}


template <int dim>
void DarcyPolynomialError<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}


template <int dim>
void DarcyPolynomialError<dim>::face(
  DoFInfo<dim>&, 
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

#endif
  
