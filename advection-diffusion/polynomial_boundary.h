/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __advectiondiffusion_polynomial_h
#define __advectiondiffusion_polynomial_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>
#include <advection-diffusion/parameters.h>
#include <integrator.h>
#include <advection-diffusion/boundary_values.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

namespace AdvectionDiffusion
{
/**
 * Provide the right hand side for a Advection-Diffusion problem with
 * polynomial solution and inhomogeneous Dirichlet boundary conditions. 
 *
 * todo: 
 * This class is matched by
 * PolynomialBoundaryError which operates on the same polynomial
 * solutions. The solution obtained is described in
 * PolynomialBoundaryError.
 * 
 * @author Guido Kanschat, Anja Bettendorf
 * @date 2014
 *
 * @ingroup integrators
 */
template <int dim>
class PolynomialBoundaryRHS : public AmandusIntegrator<dim>
{
  public:
    PolynomialBoundaryRHS(const Parameters& par,
			    const std::vector<Polynomials::Polynomial<double> > potentials_1d,
				double factor1, double factor2,
				std::vector<std::vector<double> > direction, 
				double x1, double x2, double y1, double y2);
    
    virtual void cell(DoFInfo<dim>& dinfo,
		      IntegrationInfo<dim>& info) const;
    virtual void boundary(DoFInfo<dim>& dinfo,
			  IntegrationInfo<dim>& info) const;
  private:
    dealii::SmartPointer<const Parameters, class PolynomialBoundaryRHS<dim> > parameters;
    std::vector<Polynomials::Polynomial<double> > potentials_1d;
    std::vector<std::vector<double> > direction;
    double factor1; double factor2;
    double x1; double x2; double y1; double y2;
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
class PolynomialBoundaryError : public AmandusIntegrator<dim>
{
  public:
    PolynomialBoundaryError(const Parameters& par,
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
    dealii::SmartPointer<const Parameters, class PolynomialBoundaryError<dim> > parameters;
    std::vector<Polynomials::Polynomial<double> > potentials_1d;
};

//----------------------------------------------------------------------//

template <int dim>
PolynomialBoundaryRHS<dim>::PolynomialBoundaryRHS(
  const Parameters& par,
  const std::vector<Polynomials::Polynomial<double> > potentials_1d,
  double factor1, double factor2,
  std::vector<std::vector<double> > direction, 
  double x1, double x2, double y1, double y2)
		:
		parameters(&par),
		potentials_1d(potentials_1d),
		direction(direction),
		factor1(factor1),
		factor2(factor2),
		x1(x1), 
    		x2(x2), 
   		y1(y1), 
   		y2(y2) 
{
  this->use_face = false;
  this->use_boundary = true;

}


template <int dim>
void PolynomialBoundaryRHS<dim>::cell(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{
  
  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);

  std::vector<double> px(2);
  std::vector<double> py(2);
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
   /*   const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
      potentials_1d[0].value(x, px);
      potentials_1d[0].value(y, py);
      
      rhs[k] = direction[0][0] * px[1]*py[0] + direction[1][0] * px[0]*py[1];
*/
      const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
      potentials_1d[0].value(x, px);
      potentials_1d[0].value(y, py);
      
      //rhs[k] = 0.2*x + 0.2*x*y*y + 0.4*y + 0.4*x*x*y - 8 - 4*y*y - 4*x*x;
      rhs[k] = direction[0][0]*(2*x+2*x*y*y) + direction[1][0]*(2*y+2*x*x*y) - factor1* (4+2*y*y+2*x*x);
      //rhs[k] = direction[0][0]*(2*px[1]+2*px[1]*py[0]*py[0]) + direction[1][0]*(2*py[1]+2*px[0]*px[0]*py[1]) - factor1* (4+2*py[1]*py[1]+2*px[0]*px[0]);
    }
  
  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);

}



template <int dim>
void PolynomialBoundaryRHS<dim>::boundary(
  DoFInfo<dim>& dinfo, 
  IntegrationInfo<dim>& info) const
{

    const FEValuesBase<dim> &fe = info.fe_values();
    Vector<double> &local_vector = dinfo.vector(0).block(0);

    // Boundary values step12
/*    Functions::SlitSingularityFunction<2> exact_solution;
    std::vector<double> boundary_values(fe.n_quadrature_points);
    exact_solution.value_list(fe.get_quadrature_points(), boundary_values);
*/

    // Boundary Values with function 'g' defined in boundary_values.h
    std::vector<double> g(fe.n_quadrature_points);
    static BoundaryValues<dim> boundary_function;
    boundary_function.value_list (fe.get_quadrature_points(), g);


    const unsigned int deg = fe.get_fe().tensor_degree();
    const double penalty = 2. * deg * (deg+1) * dinfo.face->measure() / dinfo.cell->measure();
    
    const std::vector<Point<dim> > &normals = fe.get_normal_vectors ();
    Point<dim> dir;
    dir(0) = direction[0][0];
    dir(1) = direction[1][0];

    std::vector<double> rhs2(info.fe_values(0).n_quadrature_points, 0.);

    std::vector<double> px(2);
    std::vector<double> py(2);
  for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
   /*   const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
      potentials_1d[0].value(x, px);
      potentials_1d[0].value(y, py);
      
      rhs[k] = direction[0][0] * px[1]*py[0] + direction[1][0] * px[0]*py[1];
*/
      const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
    //  potentials_1d[0].value(x, px);
    //  potentials_1d[0].value(y, py);
      
      //rhs[k] = 0.2*x + 0.2*x*y*y + 0.4*y + 0.4*x*x*y - 8 - 4*y*y - 4*x*x;
      //rhs2[k] = direction[0][0]*(2*x+2*x*y*y) + direction[1][0]*(2*y+2*x*x*y);
      rhs2[k] = direction[0][0]*(2*x+2*x*y*y) + direction[1][0]*(2*y+2*x*x*y) - factor1* (4+2*y*y+2*x*x);
    }


     for (unsigned k=0; k<fe.n_quadrature_points; ++k)
      {	const double dir_n=dir * normals[k];
	
	if (dir_n<0)
	    for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
	    {	
		if (x1 < dinfo.cell->center()[0] && dinfo.cell->center()[0] < x2 && 
		    y1 < dinfo.cell->center()[1] && dinfo.cell->center()[1] < y2 )
			local_vector(i) += ( (factor2*(( fe.shape_value(i,k) * penalty * g[k])
                        	  	  - (fe.normal_vector(k) * fe.shape_grad(i,k) * g[k])))
				  	  - (dir_n * g[k] * fe.shape_value(i,k)) )
                        	 	  * fe.JxW(k);
		else 
			local_vector(i) += ( (factor1*(( fe.shape_value(i,k) * penalty * g[k])
                        	  	  - (fe.normal_vector(k) * fe.shape_grad(i,k) * g[k])))
				  	  - (dir_n * g[k] * fe.shape_value(i,k)) )
                        	 	  * fe.JxW(k);
	     }	
	/*else 
	    for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
	    {	
		if (x1 < dinfo.cell->center()[0] && dinfo.cell->center()[0] < x2 && 
		    y1 < dinfo.cell->center()[1] && dinfo.cell->center()[1] < y2 )
			local_vector(i) += ( (factor2*(( fe.shape_value(i,k) * penalty * rhs2[k])
                        	  	  - (fe.normal_vector(k) * fe.shape_grad(i,k) * rhs2[k])))
				  	  - (dir_n * rhs2[k] * fe.shape_value(i,k)) )
                        	 	  * fe.JxW(k);
		else 
			local_vector(i) += ( (factor1*(( fe.shape_value(i,k) * penalty * rhs2[k])
                        	  	  - (fe.normal_vector(k) * fe.shape_grad(i,k) * rhs2[k])))
				  	  - (dir_n * rhs2[k] * fe.shape_value(i,k)) )
                        	 	  * fe.JxW(k);
	      }
	
	  { for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
	    {	
		local_vector(i) += - (1 * fe.shape_value(i,k) )
                        	 	  * fe.JxW(k);
	
	      }
	  }
	*/	
	}	

}

//----------------------------------------------------------------------//

template <int dim>
PolynomialBoundaryError<dim>::PolynomialBoundaryError(
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
void PolynomialBoundaryError<dim>::cell(
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
  
  for (unsigned int i=0;i<=2;++i)
    dinfo.value(i) = std::sqrt(dinfo.value(i));
}


//todo : include boundary values 
template <int dim>
void PolynomialBoundaryError<dim>::boundary(
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}


template <int dim>
void PolynomialBoundaryError<dim>::face(
  DoFInfo<dim>&, 
  DoFInfo<dim>&, 
  IntegrationInfo<dim>&, 
  IntegrationInfo<dim>&) const
{}

}

#endif
  
