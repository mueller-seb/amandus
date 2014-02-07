/**********************************************************************
 * $Id: laplace_polynomial.h 1394 2014-01-31 18:43:06Z kanschat $
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __laplace_polynomial_h
#define __laplace_polynomial_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

/**
 * Integrate the right hand side for a Laplace problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 */
template <int dim>
class LaplacePolynomialRHS : public LocalIntegrator<dim>
{
  public:
    LaplacePolynomialRHS(const Polynomials::Polynomial<double> curl_potential_1d,
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
 * Integrate the residual for a Laplace problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 */
template <int dim>
class LaplacePolynomialResidual : public LocalIntegrator<dim>
{
  public:
    LaplacePolynomialResidual(const Polynomials::Polynomial<double> curl_potential_1d,
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
class LaplacePolynomialError : public LocalIntegrator<dim>
{
  public:
    LaplacePolynomialError(const Polynomials::Polynomial<double> curl_potential_1d,
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
LaplacePolynomialRHS<dim>::LaplacePolynomialRHS(
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
void LaplacePolynomialRHS<dim>::cell(
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

				       // Add a gradient part to the
				       // right hand side to test for
				       // pressure
      grad_potential_1d.value(x, px);
      grad_potential_1d.value(y, py);
      rhs[0][k] += px[1]*py[0];
    }
  
  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
}


template <int dim>
void LaplacePolynomialRHS<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}


template <int dim>
void LaplacePolynomialRHS<dim>::face(
  DoFInfo<dim>&, 
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

//----------------------------------------------------------------------//

template <int dim>
LaplacePolynomialResidual<dim>::LaplacePolynomialResidual(
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
void LaplacePolynomialResidual<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  
  std::vector<std::vector<double> > rhs (1,
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

				       // Add a gradient part to the
				       // right hand side to test for
				       // pressure
      grad_potential_1d.value(x, px);
      grad_potential_1d.value(y, py);
      rhs[0][k] += px[1]*py[0];
    }
  
  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs, -1.);
  Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0),
			 info.gradients[0][0]);
}


template <int dim>
void LaplacePolynomialResidual<dim>::boundary(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<double> null(info.fe_values(0).n_quadrature_points, 0.);
  
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_residual(dinfo.vector(0).block(0), info.fe_values(0),
			    info.values[0][0],
			    info.gradients[0][0],
			    null,
			    Laplace::compute_penalty(dinfo, dinfo, deg, deg));
}


template <int dim>
void LaplacePolynomialResidual<dim>::face(
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
LaplacePolynomialError<dim>::LaplacePolynomialError(
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
void LaplacePolynomialError<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
//  Assert(dinfo.n_values() >= 4, ExcDimensionMismatch(dinfo.n_values(), 4));
}


template <int dim>
void LaplacePolynomialError<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}


template <int dim>
void LaplacePolynomialError<dim>::face(
  DoFInfo<dim>&, 
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

#endif
  
