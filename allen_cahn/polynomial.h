/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __allen_cahn_polynomial_h
#define __allen_cahn_polynomial_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>
#include <amandus/integrator.h>

namespace AllenCahn
{
using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

/**
 * Integrate the residual for a AllenCahn problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 *
 * @ingroup integrators
 */
  template <int dim>
  class PolynomialResidual : public AmandusIntegrator<dim>
  {
    public:
      PolynomialResidual(
	double diffusion,
	const Polynomials::Polynomial<double> solution_1d);
    
      virtual void cell(DoFInfo<dim>& dinfo,
			IntegrationInfo<dim>& info) const;
      virtual void boundary(DoFInfo<dim>& dinfo,
			    IntegrationInfo<dim>& info) const;
      virtual void face(DoFInfo<dim>& dinfo1,
			DoFInfo<dim>& dinfo2,
			IntegrationInfo<dim>& info1,
			IntegrationInfo<dim>& info2) const;
    private:
      double D;
      Polynomials::Polynomial<double> solution_1d;
  };


  template <int dim>
  class PolynomialError : public AmandusIntegrator<dim>
  {
    public:
      PolynomialError(const Polynomials::Polynomial<double> solution_1d);
    
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
  PolynomialResidual<dim>::PolynomialResidual(
    double diffusion,
    const Polynomials::Polynomial<double> solution_1d)
		  :
		  D(diffusion),
		  solution_1d(solution_1d)
  {
    this->use_boundary = false;
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
  
    std::vector<double> rhs (info.fe_values(0).n_quadrature_points, 0.);

    std::vector<double> px(3);
    std::vector<double> py(3);
    for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
      {
	const double x = info.fe_values(0).quadrature_point(k)(0);
	const double y = info.fe_values(0).quadrature_point(k)(1);
	solution_1d.value(x, px);
	solution_1d.value(y, py);
      
	// negative Laplacian
	rhs[k] = D*px[2]*py[0]+D*px[0]*py[2];
	// nonlinearity of true solution
	rhs[k] -= px[0]*py[0]*(px[0]*py[0]*px[0]*py[0]-1.);

	const double u = info.values[0][0][k];
	rhs[k] += u*(u*u-1.);
      }
  
    L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
    Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0),
			   info.gradients[0][0], D);
  }


  template <int dim>
  void PolynomialResidual<dim>::boundary(
    DoFInfo<dim>& dinfo, 
    IntegrationInfo<dim>& info) const
  {
    std::vector<double> null(info.fe_values(0).n_quadrature_points, 0.);
  
    const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
    Laplace::nitsche_residual(dinfo.vector(0).block(0), info.fe_values(0),
			      info.values[0][0],
			      info.gradients[0][0],
			      null,
			      Laplace::compute_penalty(dinfo, dinfo, deg, deg), D);
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
			 info1.values[0][0],
			 info1.gradients[0][0],
			 info2.values[0][0],
			 info2.gradients[0][0],
			 Laplace::compute_penalty(dinfo1, dinfo2, deg, deg), D);
  }

//----------------------------------------------------------------------//

  template <int dim>
  PolynomialError<dim>::PolynomialError(
    const Polynomials::Polynomial<double> solution_1d)
		  :
		  solution_1d(solution_1d)
  {
    this->use_boundary = false;
    this->use_face = false;
  }


  template <int dim>
  void PolynomialError<dim>::cell(
    DoFInfo<dim>& dinfo, 
    IntegrationInfo<dim>& info) const
  {
//  Assert(dinfo.n_values() >= 4, ExcDimensionMismatch(dinfo.n_values(), 4));

    std::vector<double> px(2);
    std::vector<double> py(2);
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
	unsigned int i=0;
	// 0. L^2(u)
	dinfo.value(i++) += u*u * dx;
	// 1. H^1(u)
	dinfo.value(i++) += (Du*Du) * dx;
      }
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
  
