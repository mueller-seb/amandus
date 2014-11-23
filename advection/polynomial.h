/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __advection_polynomial_h
#define __advection_polynomial_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>
#include <advection/parameters.h>
#include <integrator.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

namespace Advection
{
/**
 * Provide the right hand side for a Advection problem with
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
    dealii::SmartPointer<const Parameters, class PolynomialRHS<dim> > parameters;
    std::vector<Polynomials::Polynomial<double> > potentials_1d;
};



/**
 * Computes the error between a numerical solution and a known exact
 * polynomial solution of a Advection problem.
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
 * The according right hand sides of the Advection equations and the
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
    dealii::SmartPointer<const Parameters, class PolynomialError<dim> > parameters;
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
  this->use_boundary = false;
  this->use_face = false;
}


template <int dim>
void PolynomialRHS<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);

  std::vector<double> px(2);
  std::vector<double> py(2);
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
      const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
      potentials_1d[0].value(x, px);
      potentials_1d[0].value(y, py);
      
      //      rhs[k] = direction[0][0] * px[1]*py[0] + direction[0][1] * px[0]*py[1];
      rhs[k] = 1. * px[1]*py[0] + 2. * px[0]*py[1];
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
  this->use_boundary = false;
  this->use_face = false;
}


template <int dim>
void PolynomialError<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  Assert(dinfo.n_values() >= 2, ExcDimensionMismatch(dinfo.n_values(), 4));
  
  std::vector<double> px(3);
  std::vector<double> py(3);
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
      const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
      potentials_1d[0].value(x, px);
      potentials_1d[0].value(y, py);
      const double dx = info.fe_values(0).JxW(k);
      
      Tensor<1,dim> Du = info.gradients[0][0][k];
      Du[0] -= px[1]*py[0];
      Du[1] -= px[0]*py[1];
      double u = info.values[0][0][k];
      u -= px[0]*py[0];

      // 0. L^2(u)
      dinfo.value(0) += (u*u) * dx;
      // 1. H^1(u)
      dinfo.value(1) += (Du*Du) * dx;
    }
  
  for (unsigned int i=0;i<2;++i)
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
  
