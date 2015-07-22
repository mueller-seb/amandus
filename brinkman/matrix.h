/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __brinkman_matrix_h
#define __brinkman_matrix_h

#include <integrators/laplace.h>
#include <integrators/elasticity.h>
#include <integrators/divergence.h>
#include <integrators/l2.h>
#include <integrators/elasticity.h>

#include <integrator.h>
#include <brinkman/parameters.h>

using namespace dealii;
using namespace dealii::LocalIntegrators;

namespace Brinkman
{
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
 * @ingroup integrators
 */
  template <int dim>
  class Matrix : public AmandusIntegrator<dim>
  {
    public:
      /**
       * Constructor setting the problem parameters. The argument gets
       * stored in a SmartPointer. Thus, it has to have a longer
       * lifetime than this object.
       */
      Matrix(const Parameters& par);
    
      void cell(MeshWorker::DoFInfo<dim>& dinfo,
		typename MeshWorker::IntegrationInfo<dim>& info) const;
      void boundary(MeshWorker::DoFInfo<dim>& dinfo,
		    typename MeshWorker::IntegrationInfo<dim>& info) const;
      void face(MeshWorker::DoFInfo<dim>& dinfo1,
		MeshWorker::DoFInfo<dim>& dinfo2,
		typename MeshWorker::IntegrationInfo<dim>& info1,
		typename MeshWorker::IntegrationInfo<dim>& info2) const;
  
      void cell_error(MeshWorker::DoFInfo<dim>& dinfo,
		      typename MeshWorker::IntegrationInfo<dim>& info);

    private:
      dealii::SmartPointer<const Parameters, Matrix<dim> > parameters;
  };

  
  template <int dim>
  Matrix<dim>::Matrix(const Parameters& par)
		  : parameters(&par)
  {}

  
  template <int dim>
  void Matrix<dim>::cell(
    MeshWorker::DoFInfo<dim>& dinfo,
    typename MeshWorker::IntegrationInfo<dim>& info) const
  {
    AssertDimension (dinfo.n_matrices(),4);
    const unsigned int id = dinfo.cell->material_id();
//  deallog << id;
  
    // a(u,v)
    Elasticity::cell_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), parameters->viscosity[id]);
    L2::mass_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), parameters->resistance[id]);
    Divergence::grad_div_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), parameters->graddiv_stabilization[id]);
    // b(u,q)
    Divergence::cell_matrix(dinfo.matrix(2,false).matrix, info.fe_values(0), info.fe_values(1));
    dinfo.matrix(1,false).matrix.copy_transposed(dinfo.matrix(2,false).matrix);
  }


  template <int dim>
  void Matrix<dim>::boundary(
    MeshWorker::DoFInfo<dim>& dinfo,
    typename MeshWorker::IntegrationInfo<dim>& info) const
  {
    const unsigned int id = dinfo.cell->material_id();
    const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
    // Elasticity::nitsche_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0),
    // 			     Laplace::compute_penalty(dinfo, dinfo, deg, deg), parameters->viscosity[id]);
  }


  template <int dim>
  void Matrix<dim>::face(
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
  
    if (parameters->viscosity[id1] != 0. && parameters->viscosity[id2] != 0.)
      Elasticity::ip_matrix(dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix, 
			    dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
			    info1.fe_values(0), info2.fe_values(0),
			    Laplace::compute_penalty(dinfo1, dinfo2, deg, deg), parameters->viscosity[id1], parameters->viscosity[id2]);
  
    if (parameters->viscosity[id1] != 0. && parameters->viscosity[id2] == 0.)
      tangential_friction(dinfo1.matrix(0,false).matrix, info1.fe_values(0), parameters->saffman*std::sqrt(parameters->resistance[id2]));
  
    if (parameters->viscosity[id1] == 0. && parameters->viscosity[id2] != 0.)
      tangential_friction(dinfo2.matrix(0,false).matrix, info2.fe_values(0), parameters->saffman*std::sqrt(parameters->resistance[id1]));
  }
}


#endif
