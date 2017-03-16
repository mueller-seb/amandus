/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __elasticity_eigen_h
#define __elasticity_eigen_h

#include <amandus/integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/grad_div.h>
#include <deal.II/integrators/elasticity.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/integration_info.h>
#include <elasticity/parameters.h>
#include <amandus/elasticity/matrix_integrators.h>

#include <set>

using namespace dealii;
using namespace dealii::MeshWorker;

namespace Elasticity
{
/**
 * Integrator for Elasticity eigenvalue problems
 *
 * The finite element system is expected to consist of a vector-valued
 * divergence conforming element in first position and a discontinuous
 * scalar element in second.
 *
 * We are building a single matrix in the end, but the cell matrices come in a two-by-two block
 * structure in the order
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
class Eigen : public AmandusIntegrator<dim>
{
public:
  Eigen(const Parameters& par, const std::set<unsigned int>& dirichlet = std::set<unsigned int>());

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const;

private:
  dealii::SmartPointer<const Parameters, class Eigen<dim>> parameters;
  std::set<unsigned int> dirichlet_boundaries;
};

template <int dim>
Eigen<dim>::Eigen(const Parameters& par, const std::set<unsigned int>& dirichlet)
  : parameters(&par)
  , dirichlet_boundaries(dirichlet)
{
  // this->input_vector_names.push_back("Newton iterate");
}

template <int dim>
void
Eigen<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  if (dinfo.n_matrices() > 1)
  {
    AssertDimension(dinfo.n_matrices(), 2);
  }
  else
  {
    AssertDimension(dinfo.n_matrices(), 1);
  }

  dealii::LocalIntegrators::L2::mass_matrix(
    dinfo.matrix(0, false).matrix, info.fe_values(0));
  dealii::LocalIntegrators::Elasticity::cell_matrix(
    dinfo.matrix(0, false).matrix, info.fe_values(0), 2. * parameters->mu);
  dealii::LocalIntegrators::GradDiv::cell_matrix(
    dinfo.matrix(0, false).matrix, info.fe_values(0), parameters->lambda);

  // Eigenvalue mass matrices
  if (dinfo.n_matrices() > 1)
  {
    LocalIntegrators::L2::mass_matrix(dinfo.matrix(1, false).matrix, info.fe_values(0));
  }
}

template <int dim>
void
Eigen<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  if (dirichlet_boundaries.count(dinfo.face->boundary_id()) != 0)
  {
    dealii::LocalIntegrators::Elasticity::nitsche_matrix(
      dinfo.matrix(0, false).matrix,
      info.fe_values(0),
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2000. * parameters->mu);
    dealii::LocalIntegrators::GradDiv::nitsche_matrix(
      dinfo.matrix(0, false).matrix,
      info.fe_values(0),
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      parameters->lambda);
  }
}

template <int dim>
void
Eigen<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                 IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  dealii::LocalIntegrators::Elasticity::ip_matrix(
    dinfo1.matrix(0, false).matrix,
    dinfo1.matrix(0, true).matrix,
    dinfo2.matrix(0, true).matrix,
    dinfo2.matrix(0, false).matrix,
    info1.fe_values(0),
    info2.fe_values(0),
    dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
    2. * parameters->mu);
  dealii::LocalIntegrators::GradDiv::ip_matrix(
    dinfo1.matrix(0, false).matrix,
    dinfo1.matrix(0, true).matrix,
    dinfo2.matrix(0, true).matrix,
    dinfo2.matrix(0, false).matrix,
    info1.fe_values(0),
    info2.fe_values(0),
    dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
    parameters->lambda);
}
}

#endif
