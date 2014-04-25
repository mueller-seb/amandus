// $Id$

#ifndef __matrix_curl_curl_h
#define __matrix_curl_curl_h

#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/maxwell.h>

using namespace dealii;
using namespace LocalIntegrators;


/**
 * Integrator for curl-curl problems in weakly divergence free
 * subspace.
 *
 * The weak formulation for this system is: find \f$u,\phi\in V_h\times
 * \Psi_h\f$ such that
 * \f[
 * \begin{array}{cccclcl}\arraycolsep1pt
 *   \bigl(\mu \nabla\times u,\nabla\times v\bigr)
 *   + \bigl(\sigma u,v\bigr) &+& \bigl(v,\nabla \phi\bigr) &=&
 * \bigl(f,v) &\qquad&\forall v\in V_h
 * \\
 * \bigl(u,\nabla \psi) && &=& 0 &&\forall \psi\in\Psi_h
 * \end{array}
 * \f]
 * The finite element system is expected to consist of a vector-valued
 * curl conforming element in first position and an H<sup>1</sup>-conforming
 * scalar element in second.
 *
 * We are building a single matrix in the end, but the cell matrices come in a two-by-two block structure in the order
 * <ol>
 * <li> The curl-elliptic operator</li>
 * <li> The gradient operator</li>
 * <li> The divergence operator as transpose of the gradient</li>
 * <li> Empty</li>
 * </ol>
 */
template <int dim>
class CurlCurlMatrix : public MeshWorker::LocalIntegrator<dim>
{
public:
  /**
   * Very simple constructor, entering a single coefficient with value
   * one into #curl_coefficient and zero into #mass_coefficient, respectively.
   */
  CurlCurlMatrix();
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
		    MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const;
  
  std::vector<double> curl_coefficient;
  std::vector<double> mass_coefficient;
};


template <int dim>
CurlCurlMatrix<dim>::CurlCurlMatrix()
		:
		curl_coefficient(1, 1.),
		mass_coefficient(1, 0.)
{}


template <int dim>
void CurlCurlMatrix<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  AssertDimension (dinfo.n_matrices(), 4);
  const unsigned int id = dinfo.cell->material_id();
  AssertIndexRange (id, curl_coefficient.size());
  AssertIndexRange (id, mass_coefficient.size());
  const double mu = curl_coefficient[id];
  const double sigma = mass_coefficient[id];
  
  Maxwell::curl_curl_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), mu);
  if (sigma != 0.)
    L2::mass_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), sigma);
  Divergence::gradient_matrix(dinfo.matrix(1,false).matrix, info.fe_values(1), info.fe_values(0));
  dinfo.matrix(2,false).matrix.copy_transposed(dinfo.matrix(1,false).matrix);
}


template <int dim>
void CurlCurlMatrix<dim>::boundary(MeshWorker::DoFInfo<dim>& /*dinfo*/,
				       typename MeshWorker::IntegrationInfo<dim>& /*info*/) const
{}


template <int dim>
void CurlCurlMatrix<dim>::face(MeshWorker::DoFInfo<dim>& /*dinfo1*/,
			       MeshWorker::DoFInfo<dim>& /*dinfo2*/,
			       MeshWorker::IntegrationInfo<dim>& /*info1*/,
			       MeshWorker::IntegrationInfo<dim>& /*info2*/) const
{}


#endif