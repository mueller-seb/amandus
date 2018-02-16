/**********************************************************************
 *
 * Copyright Guido Kanschat, 2015
 *
 **********************************************************************/

#ifndef __biot_residual_h
#define __biot_residual_h

#include <amandus/biot/biot.h>
#include <deal.II/base/function.h>

using namespace dealii::MeshWorker;

namespace Biot
{

template <int dim>
class TestResidual : public Residual<dim>
{
public:
  /**
   * The constructor, storing pointers to the parameter object and
   * the function used for weak boundary values.
   */
  TestResidual(const Parameters& par, const dealii::Function<dim>& bdry, bool implicit = true);
  TestResidual(const Parameters& par, bool implicit = true);

  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;

private:
  dealii::ZeroFunction<dim> zero;

protected:
  dealii::SmartPointer<const dealii::Function<dim>, class Residual<dim>> boundary_values;
};

template <int dim>
class MandelResidual : public Residual<dim>
{
public:
  /**
   * The constructor, storing pointers to the parameter object and
   * the function used for weak boundary values.
   */
  MandelResidual(const Parameters& par, bool implicit = true);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
};

//----------------------------------------------------------------------//

template <int dim>
TestResidual<dim>::TestResidual(const Parameters& par, bool implicit)
  : Residual<dim>(par, implicit)
  , zero(2 * dim + 1)
  , boundary_values(&zero)
{
}

template <int dim>
TestResidual<dim>::TestResidual(const Parameters& par, const dealii::Function<dim>& bdry,
                                bool implicit)
  : Residual<dim>(par, implicit)
  , zero(2 * dim + 1)
  , boundary_values(&bdry)
{
}

template <int dim>
void
TestResidual<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const double factor =
    (this->timestep == 0.) ? 1. : (this->is_implicit ? this->timestep : -this->timestep);
  const double mu = factor * this->parameters->mu;

  std::vector<std::vector<double>> null(
    2 * dim + 1, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));

  //    boundary_values->vector_values(info.fe_values(0).get_quadrature_points(), null);

  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  if (dinfo.face->boundary_id() == 0 || dinfo.face->boundary_id() == 1)
    dealii::LocalIntegrators::Elasticity::nitsche_residual(
      dinfo.vector(0).block(0),
      info.fe_values(0),
      dealii::make_slice(info.values[0], 0, dim),
      dealii::make_slice(info.gradients[0], 0, dim),
      dealii::make_slice(null, 0, dim),
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * mu);
}

template <int dim>
MandelResidual<dim>::MandelResidual(const Parameters& par, bool implicit)
  : Residual<dim>(par, implicit)
{
}

template <int dim>
void
MandelResidual<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Residual<dim>::cell(dinfo, info);
}

template <int dim>
void
MandelResidual<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  //const double factor =
  //  (this->timestep == 0.) ? 1. : (this->is_implicit ? this->timestep : -this->timestep);
  //const double mu = factor * this->parameters->mu;

  std::vector<std::vector<double>> force(
    dim, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));
  for (unsigned int i = 0; i < force[1].size(); ++i)
    force[1][i] = -2.;

  //const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  if (dinfo.face->boundary_id() == 2)
  {
    dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(0), info.fe_values(0), force);
  }
}
}

#endif
