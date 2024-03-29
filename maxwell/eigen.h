/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __matrix_curl_curl_h
#define __matrix_curl_curl_h

#include <amandus/integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/maxwell.h>
#include <deal.II/meshworker/integration_info.h>

using namespace dealii;
using namespace LocalIntegrators;

/**
 * Integrators for MaxwellIntegrators equations in various forms
 */
namespace MaxwellIntegrators
{
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
 * We are building a single matrix in the end, but the cell matrices come in a two-by-two block
 * structure in the order
 * <ol>
 * <li> The curl-elliptic operator</li>
 * <li> The gradient operator</li>
 * <li> The divergence operator as transpose of the gradient</li>
 * <li> Empty</li>
 * </ol>
 */
namespace DivCurl
{
/**
 * The matrix integrator for the weakly divergence-free curl-curl
 * problem.
 *
 * @ingroup integrators
 */
template <int dim>
class Eigen : public AmandusIntegrator<dim>
{
public:
  /**
   * Very simple constructor, entering a single coefficient with value
   * one into #curl_coefficient and zero into #mass_coefficient, respectively.
   */
  Eigen();
  virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                    dealii::MeshWorker::IntegrationInfo<dim>& info) const override;
  virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const override;
  virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
                    dealii::MeshWorker::DoFInfo<dim>& dinfo2,
                    dealii::MeshWorker::IntegrationInfo<dim>& info1,
                    dealii::MeshWorker::IntegrationInfo<dim>& info2) const override;

  std::vector<double> curl_coefficient;
  std::vector<double> mass_coefficient;
};

template <int dim>
Eigen<dim>::Eigen()
  : curl_coefficient(1, 1.)
  , mass_coefficient(1, 0.)
{
  this->use_boundary = false;
  this->use_face = false;
}

template <int dim>
void
Eigen<dim>::cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                 dealii::MeshWorker::IntegrationInfo<dim>& info) const
{
  if (dinfo.n_matrices() > 4)
  {
    AssertDimension(dinfo.n_matrices(), 8);
  }
  else
  {
    AssertDimension(dinfo.n_matrices(), 4);
  }
  const unsigned int id = dinfo.cell->material_id();
  AssertIndexRange(id, curl_coefficient.size());
  AssertIndexRange(id, mass_coefficient.size());
  // const double mu = curl_coefficient[id];
  // const double sigma = mass_coefficient[id];

  Maxwell::curl_curl_matrix(dinfo.matrix(0, false).matrix, info.fe_values(0) /*, mu*/);
  // if (sigma != 0.)
  //   L2::mass_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), sigma);
  Divergence::gradient_matrix(dinfo.matrix(1, false).matrix, info.fe_values(1), info.fe_values(0));
  dinfo.matrix(2, false).matrix.copy_transposed(dinfo.matrix(1, false).matrix);
  if (dinfo.n_matrices() > 4)
  {
    L2::mass_matrix(dinfo.matrix(4, false).matrix, info.fe_values(0));
    //  dinfo.matrix(0,false).matrix.add(-8., dinfo.matrix(4,false).matrix);
  }

  //  L2::mass_matrix(dinfo.matrix(7,false).matrix, info.fe_values(1));
}

template <int dim>
void
Eigen<dim>::boundary(dealii::MeshWorker::DoFInfo<dim>& /*dinfo*/,
                     typename dealii::MeshWorker::IntegrationInfo<dim>& /*info*/) const
{
}

template <int dim>
void
Eigen<dim>::face(dealii::MeshWorker::DoFInfo<dim>& /*dinfo1*/,
                 dealii::MeshWorker::DoFInfo<dim>& /*dinfo2*/,
                 dealii::MeshWorker::IntegrationInfo<dim>& /*info1*/,
                 dealii::MeshWorker::IntegrationInfo<dim>& /*info2*/) const
{
}
}
}

#endif
