/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __advection_residual_h
#define __advection_residual_h

#include <amandus/advection/parameters.h>
#include <amandus/integrator.h>
#include <deal.II/integrators/advection.h>
#include <deal.II/integrators/l2.h>

using namespace dealii::MeshWorker;

/**
 * Local integrators for advection problem.
 */
namespace Advection
{
/**
 * Integrate the residual for an elastic problem with zero right hand side.
 *
 * @ingroup integrators
 */
template <int dim>
class Residual : public AmandusIntegrator<dim>
{
public:
  Residual(const Parameters& par, const dealii::Function<dim>& bdry);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const;

private:
  dealii::SmartPointer<const Parameters, class Residual<dim>> parameters;
  dealii::SmartPointer<const dealii::Function<dim>, class Residual<dim>> boundary_values;
};

//----------------------------------------------------------------------//

template <int dim>
Residual<dim>::Residual(const Parameters& par, const dealii::Function<dim>& bdry)
  : parameters(&par)
  , boundary_values(&bdry)
{
  this->input_vector_names.push_back("Newton iterate");
}

template <int dim>
void
Residual<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));

  const double mu = parameters->mu;
  const double lambda = parameters->lambda;

  // std::vector<std::vector<double> > null(dim,
  // std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));
  // std::fill(null[0].begin(), null[0].end(), 1);
  // dealii::LocalIntegrators::L2::L2(
  //   dinfo.vector(0).block(0), info.fe_values(0), null);

  if (parameters->linear)
  {
    dealii::LocalIntegrators::Advection::cell_residual(
      dinfo.vector(0).block(0),
      info.fe_values(0),
      dealii::make_slice(info.gradients[0], 0, dim),
      2. * mu);
    dealii::LocalIntegrators::Divergence::grad_div_residual(
      dinfo.vector(0).block(0),
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

template <int dim>
void
Residual<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  std::vector<std::vector<double>> null(
    dim, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));

  boundary_values->vector_values(info.fe_values(0).get_quadrature_points(), null);

  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  if (dinfo.face->boundary_indicator() == 0 || dinfo.face->boundary_indicator() == 1)
    dealii::LocalIntegrators::Advection::nitsche_residual(
      dinfo.vector(0).block(0),
      info.fe_values(0),
      dealii::make_slice(info.values[0], 0, dim),
      dealii::make_slice(info.gradients[0], 0, dim),
      null,
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * parameters->mu);
}

template <int dim>
void
Residual<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  dealii::LocalIntegrators::Advection::ip_residual(
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
