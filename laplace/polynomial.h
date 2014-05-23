/**********************************************************************
 * $Id$
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
#include <integrator.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

/**
 * Integrate the right hand side for a Laplace problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 *
 * @ingroup integrators
 */
template <int dim>
class LaplacePolynomialRHS : public AmandusIntegrator<dim>
{
  public:
    LaplacePolynomialRHS(const Polynomials::Polynomial<double> solution_1d);
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
    virtual void face(DoFInfo<dim>& dinfo1,
		      DoFInfo<dim>& dinfo2,
		      IntegrationInfo<dim>& info1,
		      IntegrationInfo<dim>& info2) const;
  private:
    Polynomials::Polynomial<double> solution_1d;
};


/**
 * Integrate the residual for a Laplace problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 *
 * @ingroup integrators
 */
template <int dim>
class LaplacePolynomialResidual : public AmandusIntegrator<dim>
{
  public:
    LaplacePolynomialResidual(const Polynomials::Polynomial<double> solution_1d);
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
    virtual void face(DoFInfo<dim>& dinfo1,
		      DoFInfo<dim>& dinfo2,
		      IntegrationInfo<dim>& info1,
		      IntegrationInfo<dim>& info2) const;
  private:
    Polynomials::Polynomial<double> solution_1d;
};


template <int dim>
class LaplacePolynomialError : public AmandusIntegrator<dim>
{
  public:
    LaplacePolynomialError(const Polynomials::Polynomial<double> solution_1d);
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
    virtual void face(DoFInfo<dim>& dinfo1,
		      DoFInfo<dim>& dinfo2,
		      IntegrationInfo<dim>& info1,
		      IntegrationInfo<dim>& info2) const;
  private:
    Polynomials::Polynomial<double> solution_1d;
};

//----------------------------------------------------------------------//

template <int dim>
LaplacePolynomialRHS<dim>::LaplacePolynomialRHS(
  const Polynomials::Polynomial<double> solution_1d)
		:
		solution_1d(solution_1d)
{
  this->use_boundary = false;
  this->use_face = false;
}


template <int dim>
void LaplacePolynomialRHS<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);

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
  const Polynomials::Polynomial<double> solution_1d)
		:
		solution_1d(solution_1d)
{
  this->use_boundary = true;
  this->use_face = true;
}


template <int dim>
void LaplacePolynomialResidual<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  
  std::vector<double> rhs (info.fe_values(0).n_quadrature_points, 0.);

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

  double factor = 1.;
  if (this->timestep != 0)
    {
      factor = -this->timestep;
      L2::L2(dinfo.vector(0).block(0), info.fe_values(0), info.values[0][0]);
    }
  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs, -factor);
  Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0),
			 info.gradients[0][0], factor);
}


template <int dim>
void LaplacePolynomialResidual<dim>::boundary(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  std::vector<double> null(info.fe_values(0).n_quadrature_points, 0.);
  const double factor = (this->timestep == 0.) ? 1. : this->timestep;
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_residual(dinfo.vector(0).block(0), info.fe_values(0),
			    info.values[0][0],
			    info.gradients[0][0],
			    null,
			    Laplace::compute_penalty(dinfo, dinfo, deg, deg), factor);
}


template <int dim>
void LaplacePolynomialResidual<dim>::face(
  DoFInfo<dim>& dinfo1, 
  DoFInfo<dim>& dinfo2,
  IntegrationInfo<dim>& info1, 
  IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  const double factor = (this->timestep == 0.) ? 1. : this->timestep;
  Laplace::ip_residual(dinfo1.vector(0).block(0), dinfo2.vector(0).block(0),
		  info1.fe_values(0), info2.fe_values(0),
		  info1.values[0][0],
		  info1.gradients[0][0],
		  info2.values[0][0],
		  info2.gradients[0][0],
		       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg), factor);
}

//----------------------------------------------------------------------//

template <int dim>
LaplacePolynomialError<dim>::LaplacePolynomialError(
  const Polynomials::Polynomial<double> solution_1d)
		:
		solution_1d(solution_1d)
{
  this->use_boundary = false;
  this->use_face = false;
}


template <int dim>
void LaplacePolynomialError<dim>::cell(
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
      solution_1d.value(x, px);
      solution_1d.value(y, py);
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
  
