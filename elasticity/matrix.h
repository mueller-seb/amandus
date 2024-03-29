#ifndef __elasticity_matrix_h
#define __elasticity_matrix_h

#include <amandus/elasticity/matrix_integrators.h>
#include <amandus/integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/elasticity.h>
#include <deal.II/integrators/grad_div.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/integration_info.h>
#include <elasticity/parameters.h>

#include <set>

using namespace dealii::MeshWorker;

namespace Elasticity
{
/**
 * Integrator for Elasticity problems
 *
 * The distinction between stationary and instationary problems is
 * made by the variable AmandusIntegrator::timestep, which is
 * inherited from the base class. If this variable is zero, we solve a
 * stationary problem. If it is nonzero, we assemble for an implicit
 * scheme.
 *
 * @ingroup integrators
 */
template <int dim>
class Matrix : public AmandusIntegrator<dim>
{
public:
  Matrix(const Parameters& par, const std::set<unsigned int>& dirichlet = std::set<unsigned int>(),
         const std::set<unsigned int>& tangential = std::set<unsigned int>());

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  dealii::SmartPointer<const Parameters, class Matrix<dim>> parameters;
  std::set<unsigned int> dirichlet_boundaries;
  std::set<unsigned int> tangential_boundaries;
};

template <int dim>
Matrix<dim>::Matrix(const Parameters& par, const std::set<unsigned int>& dirichlet,
                    const std::set<unsigned int>& tangential)
  : parameters(&par)
  , dirichlet_boundaries(dirichlet)
  , tangential_boundaries(tangential)
{
  // this->input_vector_names.push_back("Newton iterate");
}

template <int dim>
void
Matrix<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  AssertDimension(dinfo.n_matrices(), 1);
  if (true || parameters->linear)
  {
    dealii::LocalIntegrators::Elasticity::cell_matrix(
      dinfo.matrix(0, false).matrix, info.fe_values(0), 2. * parameters->mu);
    dealii::LocalIntegrators::GradDiv::cell_matrix(
      dinfo.matrix(0, false).matrix, info.fe_values(0), parameters->lambda);
  }
  else
  {
    Elasticity::StVenantKirchhoff::cell_matrix(dinfo.matrix(0, false).matrix,
                                               info.fe_values(0),
                                               dealii::make_slice(info.gradients[0], 0, dim),
                                               parameters->lambda,
                                               parameters->mu);
  }
}

template <int dim>
void
Matrix<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  if (dirichlet_boundaries.count(dinfo.face->boundary_id()) != 0)
  {
    dealii::LocalIntegrators::Elasticity::nitsche_matrix(
      dinfo.matrix(0, false).matrix,
      info.fe_values(0),
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * parameters->mu);
  }
  else if (tangential_boundaries.count(dinfo.face->boundary_id()) != 0)
  {
    dealii::LocalIntegrators::Elasticity::nitsche_tangential_matrix(
      dinfo.matrix(0, false).matrix,
      info.fe_values(0),
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * parameters->mu);
  }
}

template <int dim>
void
Matrix<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
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
}
}

#endif
