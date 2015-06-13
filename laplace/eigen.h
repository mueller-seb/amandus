/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __matrix_laplace_h
#define __matrix_laplace_h

#include <deal.II/meshworker/integration_info.h>
#include <integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;


/**
 * Integrators for Laplace problems.
 *
 * @ingroup integrators
 */
namespace LaplaceIntegrators
{
  template <int dim>
  class Eigen : public AmandusIntegrator<dim>
  {
    public:
      virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
			MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const;
  };
  
  
  template <int dim>
  void Eigen<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
  {
    Laplace::cell_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0));
    if(dinfo.n_matrices()==2)
      L2::mass_matrix(dinfo.matrix(1,false).matrix, info.fe_values(0));
  }
  
  
  template <int dim>
  void Eigen<dim>::boundary(
    MeshWorker::DoFInfo<dim>& dinfo,
    typename MeshWorker::IntegrationInfo<dim>& info) const
  {
    const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
    Laplace::nitsche_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0),
			    Laplace::compute_penalty(dinfo, dinfo, deg, deg));
  }
  
  
  template <int dim>
  void Eigen<dim>::face(
    MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
    MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const
  {
    const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
    Laplace::ip_matrix(dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix, 
		       dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
		       info1.fe_values(0), info2.fe_values(0),
		       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
  }
}

#endif
