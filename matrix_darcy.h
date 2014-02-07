// $Id: matrix_darcy.h 1392 2014-01-14 08:16:17Z kanschat $

#ifndef __matrix_darcy_h
#define __matrix_darcy_h

#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;


/**
 * Integrator for Darcy problems in Hdiv
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
 */
template <int dim>
class DarcyMatrix : public MeshWorker::LocalIntegrator<dim>
{
public:
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
		    MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const;
};


template <int dim>
void DarcyMatrix<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  AssertDimension (dinfo.n_matrices(), 4);
  L2::mass_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0));
  Divergence::cell_matrix(dinfo.matrix(2,false).matrix, info.fe_values(0), info.fe_values(1));
  dinfo.matrix(1,false).matrix.copy_transposed(dinfo.matrix(2,false).matrix);
}


template <int dim>
void DarcyMatrix<dim>::boundary(
  MeshWorker::DoFInfo<dim>&,
  typename MeshWorker::IntegrationInfo<dim>&) const
{}


template <int dim>
void DarcyMatrix<dim>::face(
  MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>&,
  MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>&) const
{}


#endif
