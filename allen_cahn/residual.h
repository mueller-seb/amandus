/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __allen_cahn_residual_h
#define __allen_cahn_residual_h

#include <amandus/integrator.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

namespace AllenCahn
{
using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

/**
 * Integrate the residual for a AllenCahn problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 *
 * @ingroup integrators
 */
template <int dim>
class Residual : public AmandusIntegrator<dim>
{
public:
  Residual(double diffusion);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  double D;
};

//----------------------------------------------------------------------//

template <int dim>
Residual<dim>::Residual(double diffusion)
  : D(diffusion)
{
  this->use_boundary = false;
  this->use_face = true;
}

template <int dim>
void
Residual<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));

  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
  {
    const double u = info.values[0][0][k];
    rhs[k] += u * (u * u - 1.);
  }

  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
  Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0), info.gradients[0][0], D);
}

template <int dim>
void
Residual<dim>::boundary(DoFInfo<dim>&, IntegrationInfo<dim>&) const
{
}

template <int dim>
void
Residual<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Laplace::ip_residual(dinfo1.vector(0).block(0),
                       dinfo2.vector(0).block(0),
                       info1.fe_values(0),
                       info2.fe_values(0),
                       info1.values[0][0],
                       info1.gradients[0][0],
                       info2.values[0][0],
                       info2.gradients[0][0],
                       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
                       D);
}
}

#endif
