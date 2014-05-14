// $Id$

#ifndef __matrix_heat_h
#define __matrix_heat_h

#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;


/**
 * Integrator for heat problems
 */
template <int dim>
class HeatMatrix : public MeshWorker::LocalIntegrator<dim>
{
public:
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
		    MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const;
};


template <int dim>
void HeatMatrix<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  AssertDimension (dinfo.n_matrices(), 1);
  Laplace::cell_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0));
}


template <int dim>
void HeatMatrix<dim>::boundary(
  MeshWorker::DoFInfo<dim>& dinfo,
  typename MeshWorker::IntegrationInfo<dim>& info) const
{
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0),
  			  Laplace::compute_penalty(dinfo, dinfo, deg, deg));
}


template <int dim>
void HeatMatrix<dim>::face(
  MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
  MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Laplace::ip_matrix(dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix, 
  		     dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
  		     info1.fe_values(0), info2.fe_values(0),
  		     Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
  //std::cout<<"heat matrix face"<<std::endl;
}


#endif
