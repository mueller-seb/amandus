/**********************************************************************
 * $Id: matrix_darcy_stokes.h 1369 2013-11-22 15:33:56Z kanschat $
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __matrix_darcy_stokes_h
#define __matrix_darcy_stokes_h

#include <integrators/laplace.h>
#include <integrators/elasticity.h>
#include <integrators/divergence.h>
#include <integrators/l2.h>
#include <integrators/elasticity.h>

using namespace dealii;
using namespace dealii::LocalIntegrators;

/**
 * One-sided friction interface
 * term as it shows up in the
 * Beavers-Joseph-Saffman
 * condition.
 *
 */
template <int dim>
void tangential_friction(
  FullMatrix<double>& M,
  const FEValuesBase<dim>& int_fe,
  double friction_coefficient)
{
  const unsigned int n_dofs = int_fe.dofs_per_cell;
  const unsigned int n_comp = int_fe.get_fe().n_components();
	
  AssertDimension (M.m(), n_dofs);
  AssertDimension (M.n(), n_dofs);
  AssertDimension(n_comp, dim);
	
  Point<dim> aux1;
  Point<dim> aux2;

  const double factor = friction_coefficient;
	
  for (unsigned k=0;k<int_fe.n_quadrature_points;++k)
    {
      const double dx = int_fe.JxW(k) * factor;
      const Point<dim>& n = int_fe.normal_vector(k);
      if (dim==2)
	cross_product(aux1, n);
	    
      for (unsigned i=0;i<n_dofs;++i)
	{
	  for (unsigned j=0;j<n_dofs;++j)
	    {		    
	      if (dim==2)
		for (unsigned int d=0;d<n_comp;++d)
		  M(i,j) += dx *
			    aux1(d) * int_fe.shape_value_component(i,k,d) *
			    aux1(d) * int_fe.shape_value_component(j,k,d);
	      else if (dim==3)
		{
		  Tensor<1,dim> u;
		  for (unsigned int d=0;d<dim;++d)
		    u[d] = int_fe.shape_value_component(i,k,d);
		  cross_product(aux1, u, n);
		  for (unsigned int d=0;d<dim;++d)
		    u[d] = int_fe.shape_value_component(j,k,d);
		  cross_product(aux2, u, n);
		  M(i,j) += dx * (aux1*aux2);
		}
	      else
		{
		  Assert(false, ExcNotImplemented());
		}
	    }
	}
    }
}


/**
 * Local integration functions for MeshWorker::loop().
 *
 * Assembles the bilinear form
 * @f[
 * \begin{array}{ccccl}\arraycolsep1pt
 *  a(u,v) &-& b(v,p) &=& (f,v) \\
 *  b(u,q) && &=& (g,q)
 * \end{array}
 * @f]
 * and a pressure matrix needed for preconditioning. The operators are
 * @f{eqnarray*}{
 *  a(u,v) &=& \bigl(\nu \nabla u, \nabla v\bigr)
 *    + \bigl(\rho u,v\bigr)
 *    + \bigl<\gamma\sqrt\rho u_{S,\tau},v_{S,\tau}\bigr>_{\Gamma_{SD}}
 *    + {\rm IP}
 *  \\
 *  b(u,q) &=& \bigl(\nabla\!\cdot\! u,q\bigr)
 *  \\
 *  m(p,q)  &=& \bigl(p,q\bigr)
 * @f}
 *
 * The vectors #viscosity, #resistance, and #graddiv_stabilization are indexed by the cell
 * material id and represent the coefficients in the system.
 */
template <int dim>
class DarcyStokesMatrix : public MeshWorker::LocalIntegrator<dim>
{
  public:
				     /**
				      * Default constructor, only
				      * setting the Saffman friction
				      * parameter.
				      */
    DarcyStokesMatrix();
    
				     /**
				      * Constructor, initializing
				      * default values for #viscosity
				      * and #resistance. These are
				      * zero resistance and viscosity
				      * one in material 0 (Stokes).
				      * In material 1 (Darcy), the
				      * viscosity is 0 and the
				      * resistance is the function
				      * argument.
				      */
    DarcyStokesMatrix(double resistance);
    
    void cell(MeshWorker::DoFInfo<dim>& dinfo,
	      typename MeshWorker::IntegrationInfo<dim>& info) const;
    void boundary(MeshWorker::DoFInfo<dim>& dinfo,
	      typename MeshWorker::IntegrationInfo<dim>& info) const;
    void face(MeshWorker::DoFInfo<dim>& dinfo1,
	      MeshWorker::DoFInfo<dim>& dinfo2,
	      typename MeshWorker::IntegrationInfo<dim>& info1,
	      typename MeshWorker::IntegrationInfo<dim>& info2) const;
    std::vector<double> viscosity;
    std::vector<double> resistance;
    std::vector<double> graddiv_stabilization;
    double saffman;
  
    void cell_error(MeshWorker::DoFInfo<dim>& dinfo,
		    typename MeshWorker::IntegrationInfo<dim>& info);
};


template <int dim>
DarcyStokesMatrix<dim>::DarcyStokesMatrix()
		:
		saffman(.1)
{}


template <int dim>
DarcyStokesMatrix<dim>::DarcyStokesMatrix(double res)
		:
		viscosity(2, 0.),
		resistance(2, 0.),
		graddiv_stabilization(2, 1.),
		saffman(.1)
{
  deallog << "DarcyStokes " << res << std::endl;
  viscosity[0] = 1.;
  resistance[1] = res;
}

template <int dim>
void DarcyStokesMatrix<dim>::cell(
  MeshWorker::DoFInfo<dim>& dinfo,
  typename MeshWorker::IntegrationInfo<dim>& info) const
{
  AssertDimension (dinfo.n_matrices(),4);
  const unsigned int id = dinfo.cell->material_id();
//  deallog << id;
  
				   // a(u,v)
  Elasticity::cell_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), viscosity[id]);
  L2::mass_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), resistance[id]);
  Divergence::grad_div_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), graddiv_stabilization[id]);
				   // b(u,q)
  Divergence::cell_matrix(dinfo.matrix(2,false).matrix, info.fe_values(0), info.fe_values(1));
  dinfo.matrix(1,false).matrix.copy_transposed(dinfo.matrix(2,false).matrix);
}


template <int dim>
void DarcyStokesMatrix<dim>::boundary(
  MeshWorker::DoFInfo<dim>& dinfo,
  typename MeshWorker::IntegrationInfo<dim>& info) const
{
  const unsigned int id = dinfo.cell->material_id();
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  // Elasticity::nitsche_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0),
  // 			     Laplace::compute_penalty(dinfo, dinfo, deg, deg), viscosity[id]);
}


template <int dim>
void DarcyStokesMatrix<dim>::face(
  MeshWorker::DoFInfo<dim>& dinfo1,
  MeshWorker::DoFInfo<dim>& dinfo2,
  typename MeshWorker::IntegrationInfo<dim>& info1,
  typename MeshWorker::IntegrationInfo<dim>& info2) const
{
  const unsigned int id1 = dinfo1.cell->material_id();
  const unsigned int id2 = dinfo2.cell->material_id();
				   // Here we need to implement
				   // different forms depending on
				   // whether we are in the Stokes
				   // region (interior penalty), Darcy
				   // region (zero) or at the
				   // interface (Beavers-Joseph-Saffman)
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  
  if (viscosity[id1] != 0. && viscosity[id2] != 0.)
    Elasticity::ip_matrix(dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix, 
			  dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
			  info1.fe_values(0), info2.fe_values(0),
			  Laplace::compute_penalty(dinfo1, dinfo2, deg, deg), viscosity[id1], viscosity[id2]);
  
  if (viscosity[id1] != 0. && viscosity[id2] == 0.)
    tangential_friction(dinfo1.matrix(0,false).matrix, info1.fe_values(0), saffman*std::sqrt(resistance[id2]));
  
  if (viscosity[id1] == 0. && viscosity[id2] != 0.)
    tangential_friction(dinfo2.matrix(0,false).matrix, info2.fe_values(0), saffman*std::sqrt(resistance[id1]));
}

#endif
