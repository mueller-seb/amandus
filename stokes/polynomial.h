/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __stokes_polynomial_h
#define __stokes_polynomial_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>
#include <integrator.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

namespace StokesIntegrators
{
/**
 * Provide the right hand side for a Stokes problem with polynomial
 * solution. This class is matched by StokesIntegrators::PolynomialError and
 * StokesIntegrators::PolynomialResidual, which all operate on the same polynomial
 * solutions. The solution obtained is described in
 * StokesIntegrators::PolynomialError.
 * 
 * @note No reasonable iplementation for three dimensions is provided.
 *
 * @author Guido Kanschat
 * @date 2014
 *
 * @ingroup integrators
 */
template <int dim>
class PolynomialRHS : public AmandusIntegrator<dim>
{
  public:
    PolynomialRHS(const Polynomials::Polynomial<double> curl_potential_1d,
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
 * Integrate the residual for a Stokes problem, where the solution is
 * the curl of the symmetric tensor product of a given polynomial,
 * plus the gradient of another. The solution is described in the
 * documentation of PolynomialError.
 *
 * The integration functions of this class compute the difference of
 * the corresponding function in PolynomialRHS and the weak
 * Stokes operator applied to the current solution in "Newton
 * iterate". Thus, their Frechet derivative is in the integration
 * functions of Matrix.
 *
 * @note No reasonable iplementation for three dimensions is provided.
 *
 * @author Guido Kanschat
 * @date 2014
 *
 * @ingroup integrators
 */
template <int dim>
class PolynomialResidual : public AmandusIntegrator<dim>
{
  public:
    PolynomialResidual(const Polynomials::Polynomial<double> curl_potential_1d,
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
 * Computes the error between a numerical solution and a known exact
 * polynomial solution of a Stokes problem.
 *
 * For two dimensions, the two constructor arguments are two
 * one-dimensional polynomials \f$\psi(t)\f$ and \f$\phi(t)\f$. The
 * solution to such the Stokes problem is then determined as
 *
 * \f{alignat*}{{2}
 * \mathbf u &= \nabla\times \Psi \qquad\qquad
 * & \Psi((x_1,\ldots,x_d) &= \prod \psi(x_i) \\
 * p &= \Phi
 * & \Phi((x_1,\ldots,x_d) &= \prod \phi(x_i)
 * \f}
 *
 * The according right hand sides of the Stokes equations and the
 * residuals are integrated by the functions of the classes
 * PolynomialRHS and PolynomialResidual.
 *
 * If computing on a square, say \f$[-1,1]^2\f$, the boundary
 * conditions of the Stokes problem are determined as follows: if
 * \f$\psi(-1)=\psi(1)=0\f$, the velocity has the boundary condition
 * \f$\mathbf u\cdot \mathbf n=0\f$ (slip). If in addition
 * \f$\psi'(-1)=\psi'(1)=0\f$, then there holds on the boundary
 * \f$\mathbf u=0\f$ (no-slip).
 *
 * @note No reasonable iplementation for three dimensions is provided.
 *
 * @author Guido Kanschat
 * @date 2014
 *
 * @ingroup integrators
 */
template <int dim>
class PolynomialError : public AmandusIntegrator<dim>
{
  public:
    PolynomialError(const Polynomials::Polynomial<double> curl_potential_1d,
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
PolynomialRHS<dim>::PolynomialRHS(
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
void PolynomialRHS<dim>::cell(
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
void PolynomialRHS<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}


template <int dim>
void PolynomialRHS<dim>::face(
  DoFInfo<dim>&, 
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

//----------------------------------------------------------------------//

template <int dim>
PolynomialResidual<dim>::PolynomialResidual(
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
void PolynomialResidual<dim>::cell(
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
  Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0),
			 make_slice(info.gradients[0], 0, dim));
  Divergence::gradient_residual(dinfo.vector(0).block(0), info.fe_values(0),
  				info.values[0][dim], -1.);
  Divergence::cell_residual(dinfo.vector(0).block(1), info.fe_values(1),
  			    make_slice(info.gradients[0], 0, dim), 1.);
}


template <int dim>
void PolynomialResidual<dim>::boundary(
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
void PolynomialResidual<dim>::face(
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

//----------------------------------------------------------------------//

template <int dim>
PolynomialError<dim>::PolynomialError(
  const Polynomials::Polynomial<double> curl_potential_1d,
  const Polynomials::Polynomial<double> grad_potential_1d)
		:
		curl_potential_1d(curl_potential_1d),
		grad_potential_1d(grad_potential_1d)
{
  this->num_errors = 5;
  this->use_boundary = false;
  this->use_face = false;
}


template <int dim>
void PolynomialError<dim>::cell(
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
      // 0. L^2(u)
      dinfo.value(0) += (u0*u0+u1*u1) * dx;
      // 1. H^1(u)
      dinfo.value(1) += ((Du0*Du0)+(Du1*Du1)) * dx;
      // 2. div u
      dinfo.value(2) =
	Divergence::norm(info.fe_values(0), make_slice(info.gradients[0], 0, dim));
      // 3. L^2(p) up to mean value
      dinfo.value(3) += p*p * dx;
      // 4. H^1(p)
      dinfo.value(4) += Dp*Dp * dx;
    }
  
  for (unsigned int i=0;i<=4;++i)
    dinfo.value(i) = std::sqrt(dinfo.value(i));
}


template <int dim>
void PolynomialError<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}


template <int dim>
void PolynomialError<dim>::face(
  DoFInfo<dim>&, 
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}
}

#endif
