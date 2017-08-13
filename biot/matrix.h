#ifndef __biot_matrix_h
#define __biot_matrix_h

#include <amandus/biot/biot.h>

using namespace dealii::MeshWorker;

/**
 * Namespace for discretization of Biot's law.
 *
 * The solution components are
 * <ol>
 * <li> The solid displacement</li>
 * <li> The fluid velocity</li>
 * <li> The fluid pressure</li>
 * </ol>
 */
namespace Biot
{
/**
 * Integrator for Biot problems.
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
class TestMatrix : public Matrix<dim>
{
public:
  TestMatrix(const Parameters& par)
    : Matrix<dim>(par)
  {
  }

  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
};

template <int dim>
void
TestMatrix<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const double factor = (this->timestep == 0.) ? 1. : this->timestep;
  const double mu = factor * this->parameters->mu;

  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  // Elasticity
  if (dinfo.face->boundary_id() == 0 || dinfo.face->boundary_id() == 1)
    dealii::LocalIntegrators::Elasticity::nitsche_matrix(
      dinfo.matrix(0, false).matrix,
      info.fe_values(0),
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * mu);
  // Darcy has no boundary terms
}
}

#endif
