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

namespace Elasticity
{
/**
 * Provide the right hand side for a Elasticity problem with
 * polynomial solution. This class is matched by
 * PolynomialError which operates on the same polynomial
 * solutions. The solution obtained is described in
 * PolynomialError.
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
    PolynomialRHS(const Parameters& par,
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
 * Since we are planning to use this even in the nearly incompressible
 * case, we use Helmholtz decomposition and represent the solution as
 * the sum of the gradient of one polynomial and the curl of either
 * one (id 2D) or three (in 3D) polynomials. These are in the vector
 * of polynomials $\phi$ given to the constructor, such that the gradient
 * potential is first.
 *
 * \f{alignat*}{{2}
 * \mathbf u &= \nabla \phi_0 + \nabla\times \phi_1 & \text{2D} \\
 * \mathbf u &= \nabla \phi_0 + \nabla\times (\phi_1,\dots,\phi_3)^T & \text{3D} \\
 * \f}
 *
 * The according right hand sides of the Elasticity equations and the
 * residuals are integrated by the functions of the classes
 * PolynomialRHS.
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
    PolynomialError(const Parameters& par,
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
PolynomialRHS<dim>::PolynomialRHS(
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
void PolynomialRHS<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<std::vector<double> > rhs (dim,
					 std::vector<double>(info.fe_values(0).n_quadrature_points));
  std::vector<std::vector<double> > phi(dim, std::vector<double>(4));
  
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
      const Point<dim>& x = info.fe_values(0).quadrature_point(k);
      Tensor<1,dim> DivDu;
      Tensor<1,dim> DDivu;
      
      if (dim == 2)
	{
	  // The gradient potential
	  potentials_1d[0].value(x(0), phi[0]);
	  potentials_1d[0].value(x(1), phi[1]);
	  DivDu[0] -= phi[0][3]*phi[1][0] + 0.5 * (phi[0][1]*phi[1][2] + phi[0][1]*phi[1][2]);
	  DivDu[1] -= phi[0][0]*phi[1][3] + 0.5 * (phi[0][2]*phi[1][1] + phi[0][2]*phi[1][1]);
	  DDivu[0] -= phi[0][2]*phi[1][0] + phi[0][1]*phi[1][1];
	  DDivu[1] -= phi[0][1]*phi[1][1] + phi[0][0]*phi[1][2];
	  // The curl potential
	  potentials_1d[1].value(x(0), phi[0]);
	  potentials_1d[1].value(x(1), phi[1]);
	  // div epsilon(curl) = Delta?
	  DivDu[0] -= -phi[0][2]*phi[1][1] + 0.5 * (phi[0][1]*phi[1][2] - phi[0][1]*phi[1][2]);
	  DivDu[1] -=  phi[0][1]*phi[1][2] + 0.5 * (phi[0][2]*phi[1][1] - phi[0][2]*phi[1][1]);
	  // div curl = 0
	}

      for (unsigned int d=0;d<dim;++d)
	rhs[d][k] = 2. * parameters->mu * DivDu[d] + parameters->lambda * DDivu[d];
    }
  
  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
}


template <int dim>
void PolynomialRHS<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

//----------------------------------------------------------------------//

template <int dim>
PolynomialError<dim>::PolynomialError(
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
void PolynomialError<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
//  Assert(dinfo.n_values() >= 4, ExcDimensionMismatch(dinfo.n_values(), 4));
  
  std::vector<std::vector<double> > phi(dim, std::vector<double>(3));
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
      
      potentials_1d[0].value(x(0), phi[0]);
      potentials_1d[0].value(x(1), phi[1]);
      u[0] -= phi[0][1]*phi[1][0];
      u[1] -= phi[0][0]*phi[1][1];
      Du[0][0] -= phi[0][2]*phi[1][0];
      Du[0][1] -= phi[0][1]*phi[1][1];
      Du[1][0] -= phi[0][1]*phi[1][1];
      Du[1][1] -= phi[0][0]*phi[1][2];
      
      if (dim == 2)
	{
	  potentials_1d[1].value(x(0), phi[0]);
	  potentials_1d[1].value(x(1), phi[1]);
	  
	  u[0] += phi[0][0]*phi[1][1];
	  u[1] -= phi[0][1]*phi[1][0];
	  Du[0][0] += phi[0][1]*phi[1][1];
	  Du[0][1] += phi[0][0]*phi[1][2];
	  Du[1][0] -= phi[0][2]*phi[1][0];
	  Du[1][1] -= phi[0][1]*phi[1][1];
	}

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
  
