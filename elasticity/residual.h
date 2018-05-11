/**********************************************************************
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __elasticity_residual_h
#define __elasticity_residual_h

#include <amandus/elasticity/integrators.h>
#include <amandus/integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/elasticity.h>
#include <deal.II/integrators/grad_div.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <elasticity/parameters.h>

using namespace dealii::MeshWorker;

/**
 * Discretization of elastic deformation. Formulation of stress and
 * strain relationship follows the book of Ciarlet (Volume I).
 *
 * Here a list of quantities of the deformation \f$ x \mapsto \phi(x) \f$
 * and their meaning:
 * <dl>
 * <dt>\f$ u(x) = \phi(x)-x \f$</dt> <dd>Displacement</dd>
 * <dt> \f$ F(x) = \nabla \phi(x) = I + \nabla u(x) \f$</dt>
 * <dd>Deformation gradient</dd>
 * <dt>\f$ C = F^TF = I + \nabla u + (\nabla u)^T
 * + (\nabla u)^T(\nabla u)\f$</dt><dd>The right Cauchy-Green strain
 * tensor</dd>
 * <dt>\f$ E = \tfrac12 (C-I) = \tfrac12\bigl(\nabla u + (\nabla u)^T
 * + (\nabla u)^T(\nabla u)\bigr)\f$</dt><dd>The Green-St. Venant strain
 * tensor</dd>
 * <dt>\f$ J = \operatorname{det} F \f$</dt><dd>The deformed volume element</dd>
 *</dl>
 *
 * The quantity describing the forces in an elastically deformed body
 * is the stress. Depending on whether the stress is measured in
 * deformed or undeformed coordinates, we have the Cauchy or the two
 * Piola-Kirchhoff stress tensors. Material laws are typically given
 * as either the Cauchy stress \f$\hat T\f$ or the second Piola-Kirchhoff
 * stress tensor \f$ \Sigma = J F^{-1} \hat T F^{-T}\f$. Using the
 * second, we can write the equations of nonlinear elasticity in weak
 * form as
 * \f[
 * \int_\Omega \bigl((I+F) \Sigma\bigr) \colon \nabla v \,dx =
 * \int_\Omega f \cdot v \,dx +
 * \int_{\Gamma_N} \sigma_n \cdot v \,ds.
 * \f]
 *
 * This equation is complemented by a stress-strain relation \f$\Sigma
 * = \Sigma(E)\f$ depending on the material, for instance Hooke's law
 * (with Lamé-Navier coefficients)
 * \f[
 * \Sigma = \lambda (\operatorname{tr} E) I + 2 \mu E.
 * \f]
 *
 * The model is called <b>geometrically linear</b> if in this equation
 * \f$F\f$ is replaced by zero, and the quadratic term in the
 * Green-St. Venant stress tensor is neglected. Combining this with
 * Hooke's law and the fact that \f$\tfrac12(A+A^T)\f$ is the
 * projection of a matrix \f$A\f$ to the subspace of symmetric
 * matrices and \f$ (\operatorname{tr} A) I \f$ its projection on the
 * subspace spanned by the identity, we obtain the Lamé-Navier
 * equations
 * \f[
 * 2\mu \int_\Omega \epsilon(u):\epsilon(v) \,dx +
 * \lambda \int_\Omega \nabla\!\cdot\!u \nabla\!\cdot\!v \,dx =
 * \int_\Omega f \cdot v \,dx +
 * \int_{\Gamma_N} \sigma_n \cdot v \,ds.
 * \f]
 *
 * The local integrators for this equation a re part of the deal.II
 * library. The integrators in the namespace StVenantKirchhoff extend
 * this to the geometrically nonlinear weak formulation above, but
 * with the linear stress-strain relation established by Hooke's law.
 *
 */
namespace Elasticity
{
/**
 * Integrate the residual for an elastic problem with right hand side
 * depending on the second template parameter. Possible values are
 * zero for no force and one for gravity (constant force strength
 * -9.80665 in last coordinate direction)
 *
 * @ingroup integrators
 */
template <int dim, int force_type = 0>
class Residual : public AmandusIntegrator<dim>
{
public:
  Residual(const Parameters& par, const dealii::Function<dim>& bdry,
           const std::set<unsigned int>& dirichlet = std::set<unsigned int>());

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  dealii::SmartPointer<const Parameters, class Residual<dim>> parameters;
  dealii::SmartPointer<const dealii::Function<dim>, class Residual<dim>> boundary_values;
  std::set<unsigned int> dirichlet_boundaries;
};

//----------------------------------------------------------------------//

template <int dim, int force_type>
Residual<dim, force_type>::Residual(const Parameters& par, const dealii::Function<dim>& bdry,
                                    const std::set<unsigned int>& dirichlet)
  : parameters(&par)
  , boundary_values(&bdry)
  , dirichlet_boundaries(dirichlet)
{
  this->input_vector_names.push_back("Newton iterate");
}

template <int dim, int force_type>
void
Residual<dim, force_type>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));

  const double mu = parameters->mu;
  const double lambda = parameters->lambda;

  if (force_type == 1)
  {
    std::vector<std::vector<double>> force(
      dim, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));
    std::fill(force[dim - 1].begin(), force[dim - 1].end(), 9.80665);
    dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(0), info.fe_values(0), force);
  }

  if (parameters->linear)
  {
    dealii::LocalIntegrators::Elasticity::cell_residual(
      dinfo.vector(0).block(0),
      info.fe_values(0),
      dealii::make_slice(info.gradients[0], 0, dim),
      2. * mu);
    dealii::LocalIntegrators::GradDiv::cell_residual(dinfo.vector(0).block(0),
                                                     info.fe_values(0),
                                                     dealii::make_slice(info.gradients[0], 0, dim),
                                                     lambda);
  }
  else
  {
    StVenantKirchhoff::cell_residual(dinfo.vector(0).block(0),
                                     info.fe_values(0),
                                     dealii::make_slice(info.gradients[0], 0, dim),
                                     lambda,
                                     mu);
  }
}

template <int dim, int force_type>
void
Residual<dim, force_type>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  std::vector<std::vector<double>> null(
    dim, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));

  boundary_values->vector_values(info.fe_values(0).get_quadrature_points(), null);

  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  if (dirichlet_boundaries.count(dinfo.face->boundary_id()) != 0)
  {
    dealii::LocalIntegrators::Elasticity::nitsche_residual(
      dinfo.vector(0).block(0),
      info.fe_values(0),
      dealii::make_slice(info.values[0], 0, dim),
      dealii::make_slice(info.gradients[0], 0, dim),
      null,
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * parameters->mu);
  }
}

template <int dim, int force_type>
void
Residual<dim, force_type>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2,
                                IntegrationInfo<dim>& info1, IntegrationInfo<dim>& info2) const
{
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
    2. * parameters->mu);
}
}

#endif
