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
#include <amandus/advection-diffusion/parameters.h>
#include <amandus/integrator.h>
#include <amandus/advection-diffusion/boundary_values.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

namespace AdvectionDiffusion
{
/**
 * Provide the right hand side for a Advection-Diffusion problem with
 * polynomial solution and inhomogeneous Dirichlet boundary conditions. 
 *
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
  this->use_boundary = false;

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
      const double x = info.fe_values(0).quadrature_point(k)(0);
      const double y = info.fe_values(0).quadrature_point(k)(1);
      potentials_1d[0].value(x, px);
      potentials_1d[0].value(y, py);
      
      // different right-hand-sides:
      rhs[k] = 1;
      //rhs[k]= x;   
      //rhs[k] = 0.2*x + 0.2*x*y*y + 0.4*y + 0.4*x*x*y - 8 - 4*y*y - 4*x*x;
      //rhs[k] = direction[0][0]*(2*x+2*x*y*y) + direction[1][0]*(2*y+2*x*x*y) - factor1* (4+2*y*y+2*x*x);
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


     for (unsigned k=0; k<fe.n_quadrature_points; ++k)
      {	const double dir_n=dir * normals[k];
	
	// Boundary condition holds at the inflow-boundary:
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
		
	}	

}

}

#endif
  
