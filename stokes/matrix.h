// $Id$

#ifndef __matrix_stokes_h
#define __matrix_stokes_h

#include <deal.II/meshworker/integration_info.h>
#include <integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;


/**
 * Integrator for Stokes problems in Hdiv
 *
 * The finite element system is expected to consist of a vector-valued
 * divergence conforming element in first position and a discontinuous
 * scalar element in second.
 *
 * We are building a single matrix in the end, but the cell matrices come in a two-by-two block structure in the order
 * <ol>
 * <li> The vector-valued Laplacian</li>
 * <li> The gradient operator</li>
 * <li> The divergence operator as transpose of the gradient</li>
 * <li> Empty</li>
 * </ol>
 *
 * @ingroup integrators
 */
template <int dim>
class StokesMatrix : public AmandusIntegrator<dim>
{
public:
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
		    MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const;
};


template <int dim>
void StokesMatrix<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  AssertDimension (dinfo.n_matrices(), 4);
  Laplace::cell_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0));
  Divergence::cell_matrix(dinfo.matrix(2,false).matrix, info.fe_values(0), info.fe_values(1));
  dinfo.matrix(1,false).matrix.copy_transposed(dinfo.matrix(2,false).matrix);
}


template <int dim>
void StokesMatrix<dim>::boundary(
  MeshWorker::DoFInfo<dim>& dinfo,
  typename MeshWorker::IntegrationInfo<dim>& info) const
{
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0),
  			  Laplace::compute_penalty(dinfo, dinfo, deg, deg));
}


template <int dim>
void StokesMatrix<dim>::face(
  MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
  MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Laplace::ip_matrix(dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix, 
  		     dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
  		     info1.fe_values(0), info2.fe_values(0),
  		     Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
}


#endif
