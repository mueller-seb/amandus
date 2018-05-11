#ifndef __biot_biot_h
#define __biot_biot_h

#include <amandus/integrator.h>
#include <biot/parameters.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/elasticity.h>
#include <deal.II/integrators/grad_div.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/integration_info.h>

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
constexpr int
pos(const unsigned int i, const unsigned int j)
{
  return 3 * i + j;
}

/**
* Matrix integrator for Biot problems.
*
* This class implements the cell and interior face terms for Biot's
* problem. The boundary terms have to be implemented by a derived
* class, since they depend on the actual problem setting.
*
* The distinction between stationary and instationary problems is
* made by the variable AmandusIntegrator::timestep, which is
* inherited from the base class. If this variable is zero, we solve a
* stationary problem. If it is nonzero, we assemble for an implicit
* scheme.
*
* The structure of the matrix is
* \f[
* \begin{pmatrix}
* \nabla\cdot(2\mu\epsilon + \lambda I \nabla \cdot)
* \end{pmatrix}
* \f]
*
* @ingroup integrators
*/
template <int dim>
class Matrix : public AmandusIntegrator<dim>
{
public:
  Matrix(const Parameters& par);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

protected:
  dealii::SmartPointer<const Parameters, class Matrix<dim>> parameters;
};

/**
 * Integrate the residual for a Biot problem with zero right hand
 * side.
 *
 * Inhomogeneous problems and boundary conditions must be
 * implemented by a derived class.
 *
 * @ingroup integrators
 */
template <int dim>
class Residual : public AmandusIntegrator<dim>
{
public:
  /**
   * The constructor, storing pointers to the parameter object and
   * the function used for weak boundary values.
   */
  Residual(const Parameters& par, bool implicit = true);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

protected:
  dealii::SmartPointer<const Parameters, class Residual<dim>> parameters;
  bool is_implicit;
};

template <int dim>
Matrix<dim>::Matrix(const Parameters& par)
  : parameters(&par)
{
}

template <int dim>
void
Matrix<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const double factor = (this->timestepping) ? this->timestep : 1.;
  const double mu = factor * parameters->mu;
  const double lambda = factor * parameters->lambda;
  const double resistance = factor * parameters->resistance;
  //    const double p_to_disp = factor * parameters->p_to_disp;

  AssertDimension(dinfo.n_matrices(), 9);

  // Timestepping for pressure and dilation
  if (this->timestepping)
  {
    dealii::LocalIntegrators::L2::mass_matrix(
      dinfo.matrix(pos(2, 2), false).matrix, info.fe_values(1), parameters->storage);
    dealii::LocalIntegrators::Divergence::cell_matrix(dinfo.matrix(pos(2, 0), false).matrix,
                                                      info.fe_values(0),
                                                      info.fe_values(1),
                                                      -parameters->p_to_disp);
  }

  // Elasticity
  dealii::LocalIntegrators::Elasticity::cell_matrix(
    dinfo.matrix(pos(0, 0), false).matrix, info.fe_values(0), 2. * mu);
  dealii::LocalIntegrators::GradDiv::cell_matrix(
    dinfo.matrix(pos(0, 0), false).matrix, info.fe_values(0), lambda);

  // Darcy
  dealii::LocalIntegrators::L2::mass_matrix(
    dinfo.matrix(pos(1, 1), false).matrix, info.fe_values(0), resistance);
  dealii::LocalIntegrators::Divergence::cell_matrix(
    dinfo.matrix(pos(2, 1), false).matrix, info.fe_values(0), info.fe_values(1), -factor);
  dinfo.matrix(pos(1, 2), false).matrix.copy_transposed(dinfo.matrix(pos(2, 1), false).matrix);
  dinfo.matrix(pos(1, 2), false).matrix *= -1.;

  // Coupling from Darcy to Elasticity
  dinfo.matrix(pos(0, 2), false).matrix.copy_transposed(dinfo.matrix(pos(2, 1), false).matrix);
  dinfo.matrix(pos(0, 2), false).matrix *=
    -parameters->p_to_disp; // * (2.*parameters->mu + parameters->lambda);
}

template <int dim>
void
Matrix<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                  IntegrationInfo<dim>& info2) const
{
  const double factor = (this->timestepping) ? this->timestep : 1.;
  const double mu = factor * parameters->mu;

  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  // Elasticity
  dealii::LocalIntegrators::Elasticity::ip_matrix(
    dinfo1.matrix(0, false).matrix,
    dinfo1.matrix(0, true).matrix,
    dinfo2.matrix(0, true).matrix,
    dinfo2.matrix(0, false).matrix,
    info1.fe_values(0),
    info2.fe_values(0),
    dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
    2. * mu);
  // grad div ip is missing for full DG.

  // Darcy has no interior penalty
}

template <int dim>
Residual<dim>::Residual(const Parameters& par, bool implicit)
  : parameters(&par)
  , is_implicit(implicit)
{
}

template <int dim>
void
Residual<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));

  const double factor =
    (this->timestepping) ? (is_implicit ? this->timestep : -this->timestep) : 1.;
  const double mu = factor * parameters->mu;
  const double lambda = factor * parameters->lambda;
  const double resistance = factor * parameters->resistance;
  const double p_to_disp = factor * parameters->p_to_disp;

  // Timestepping for pressure and dilation
  if (this->timestepping)
  {
    dealii::LocalIntegrators::L2::L2(
      dinfo.vector(0).block(2), info.fe_values(1), info.values[0][2 * dim], parameters->storage);
    dealii::LocalIntegrators::Divergence::cell_residual(
      dinfo.vector(0).block(2),
      info.fe_values(1),
      dealii::make_slice(info.gradients[0], 0, dim),
      -parameters->p_to_disp);
  }

  // Elasticity
  dealii::LocalIntegrators::Elasticity::cell_residual(dinfo.vector(0).block(0),
                                                      info.fe_values(0),
                                                      dealii::make_slice(info.gradients[0], 0, dim),
                                                      2. * mu);
  dealii::LocalIntegrators::GradDiv::cell_residual(dinfo.vector(0).block(0),
                                                   info.fe_values(0),
                                                   dealii::make_slice(info.gradients[0], 0, dim),
                                                   lambda);
  // Darcy
  dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(1),
                                   info.fe_values(0),
                                   dealii::make_slice(info.values[0], dim, dim),
                                   resistance);
  dealii::LocalIntegrators::Divergence::gradient_residual(
    dinfo.vector(0).block(1), info.fe_values(0), info.values[0][2 * dim], -factor);
  dealii::LocalIntegrators::Divergence::cell_residual(
    dinfo.vector(0).block(2),
    info.fe_values(1),
    dealii::make_slice(info.gradients[0], dim, dim),
    -factor);
  // Coupling from Darcy to Elasticity
  dealii::LocalIntegrators::Divergence::gradient_residual(
    dinfo.vector(0).block(0), info.fe_values(0), info.values[0][2 * dim], -p_to_disp);
}

template <int dim>
void
Residual<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const
{
  if (info1.values.size() == 0)
    return;
  const double factor =
    (this->timestepping) ? (is_implicit ? this->timestep : -this->timestep) : 1.;
  const double mu = factor * parameters->mu;

  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  dealii::LocalIntegrators::Elasticity::ip_residual(
    dinfo1.vector(0).block(0),
    dinfo2.vector(0).block(0),
    info1.fe_values(0),
    info2.fe_values(0),
    dealii::make_slice(info1.values[0], 0, dim),
    dealii::make_slice(info1.gradients[0], 0, dim),
    dealii::make_slice(info2.values[0], 0, dim),
    dealii::make_slice(info2.gradients[0], 0, dim),
    dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
    2. * mu);
}
}

#endif
