/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __elasticity_polynomial_h
#define __elasticity_polynomial_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>
#include <elasticity/parameters.h>
#include <integrator.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

/**
 * Provide the right hand side for a Elasticity problem with
 * polynomial solution. This class is matched by
 * ElasticityPolynomialError which operates on the same polynomial
 * solutions. The solution obtained is described in
 * ElasticityPolynomialError.
 * 
 * @author Guido Kanschat
 * @date 2014
 *
 * @ingroup integrators
 */
template <int dim>
class ElasticityPolynomialRHS : public AmandusIntegrator<dim>
{
  public:
    ElasticityPolynomialRHS(const Parameters& par,
			    const std::vector<Polynomials::Polynomial<double> > potentials_1d);
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
  private:
    dealii::SmartPointer<const Parameters, class Residual<dim> > parameters;
    std::vector<Polynomials::Polynomial<double> > potentials_1d;
};



/**
 * Computes the error between a numerical solution and a known exact
 * polynomial solution of a Elasticity problem.
 *
 * For two dimensions, the two constructor arguments are two
 * one-dimensional polynomials \f$\psi(t)\f$ and \f$\phi(t)\f$. The
 * solution to such the Elasticity problem is then determined as
 *
 * \f{alignat*}{{2}
 * \mathbf u &= \nabla\times \Psi \qquad\qquad
 * & \Psi((x_1,\ldots,x_d) &= \prod \psi(x_i) \\
 * p &= \Phi
 * & \Phi((x_1,\ldots,x_d) &= \prod \phi(x_i)
 * \f}
 *
 * The according right hand sides of the Elasticity equations and the
 * residuals are integrated by the functions of the classes
 * ElasticityPolynomialRHS and ElasticityPolynomialResidual.
 *
 * If computing on a square, say \f$[-1,1]^2\f$, the boundary
 * conditions of the Elasticity problem are determined as follows: if
 * \f$\psi(-1)=\psi(1)=0\f$, the velocity has the boundary condition
 * \f$\mathbf u\cdot \mathbf n=0\f$ (slip). If in addition
 * \f$\psi'(-1)=\psi'(1)=0\f$, then there holds on the boundary
 * \f$\mathbf u=0\f$ (no-slip).
 *
 * @author Guido Kanschat
 * @date 2014
 *
 * @ingroup integrators
 */
template <int dim>
class ElasticityPolynomialError : public AmandusIntegrator<dim>
{
  public:
    ElasticityPolynomialError(const Parameters& par,
			      const std::vector<Polynomials::Polynomial<double> > potentials_1d);
    
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
    std::vector<Polynomials::Polynomial<double> > potentials_1d;
};

//----------------------------------------------------------------------//

template <int dim>
ElasticityPolynomialRHS<dim>::ElasticityPolynomialRHS(
  const Parameters& par,
  const std::vector<Polynomials::Polynomial<double> > potentials_1d)
		:
		parameters(&par),
		potentials_1d(potentials_1d)
{
  AssertDimension(potentials_1d.size(), (dim==2 ? 2 : dim+1));
  this->use_boundary = false;
  this->use_face = false;
}


template <int dim>
void ElasticityPolynomialRHS<dim>::cell(
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

      if (dim==2)
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
void ElasticityPolynomialRHS<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

//----------------------------------------------------------------------//

template <int dim>
StokesPolynomialError<dim>::StokesPolynomialError(
  const Parameters& par,
  const std::vector<Polynomials::Polynomial<double> > potentials_1d)
		:
		parameters(&par),
		potentials_1d(potentials_1d)
{
  AssertDimension(potentials_1d.size(), (dim==2 ? 2 : dim+1));
  this->use_boundary = false;
  this->use_face = false;
}


template <int dim>
void StokesPolynomialError<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
//  Assert(dinfo.n_values() >= 4, ExcDimensionMismatch(dinfo.n_values(), 4));
  
  std::vector<std::vector<double> > p(dim, std::vector<double>(3));
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
      const Point<dim>& x = info.fe_values(0).quadrature_point(k);
      
      double u[dim];
      Tensor<1,dim> Du[dim];
      const double dx = info.fe_values(0).JxW(k);

      for (unsigned int d=0;d<dim;++d)
	{
	  u[d] = info.values[0][d][k];
	  Du[d] = info.gradients[0][d][k];
	}
      
      if (dim = 2)
	{
	  potentials_1d[1].value(x(0), p[0]);
	  potentials_1d[1].value(x(1), p[1]);
	  
	  Du[0][0] -= p[0][1]*p[1][1];
	  Du[0][1] -= p[0][0]*p[1][2];
	  Du[1][0] += p[0][2]*p[1][0];
	  Du[1][1] += p[0][1]*p[1][1];
	  u[0] -= p[0][0]*p[1][1];
	  u[1] += p[0][1]*p[1][0];
	}
      
      potentials[0]_1d.value(x(0), p[0]);
      potentials[0]_1d.value(x(1), p[1]);
      u[0] += p[0][1]*p[1][0];
      u[1] += p[0][0]*p[1][1];

      double uu = 0., Duu= 0., div = 0.;
      for (unsigned int d=0;d<dim;++d)
	{
	  uu += u[d]*u[d];
	  Duu += Du[d]*Du[d];
	  div += Du[d][d]*Du[d][d];
	}
      
      // 0. L^2(u)
      dinfo.value(0) += uu * dx;
      // 1. H^1(u)
      dinfo.value(1) += Duu * dx;
      // 2. div u
      dinfo.value(2) =
	Divergence::norm(info.fe_values(0), make_slice(info.gradients[0], 0, dim));
    }
  
  for (unsigned int i=0;i<=4;++i)
    dinfo.value(i) = std::sqrt(dinfo.value(i));
}


template <int dim>
void StokesPolynomialError<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}


template <int dim>
void StokesPolynomialError<dim>::face(
  DoFInfo<dim>&, 
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

#endif
  
