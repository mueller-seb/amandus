/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __laplace_polynomial_h
#define __laplace_polynomial_h

#include <amandus/integrator.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

namespace LaplaceIntegrators
{
/**
 * Integrate the right hand side for a Laplace problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 *
 * @ingroup integrators
 */
template <int dim>
class PolynomialRHS : public AmandusIntegrator<dim>
{
public:
  PolynomialRHS(const Polynomials::Polynomial<double> solution_1d);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  Polynomials::Polynomial<double> solution_1d;
};

/**
 * Integrate the residual for a Laplace problem, where the
 * solution is the curl of the symmetric tensor product of a given
 * polynomial, plus the gradient of another.
 *
 * @ingroup integrators
 */
template <int dim>
class PolynomialResidual : public AmandusIntegrator<dim>
{
public:
  PolynomialResidual(const Polynomials::Polynomial<double> solution_1d);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  Polynomials::Polynomial<double> solution_1d;
};

template <int dim>
class PolynomialBoundary : public Function<dim>
{
public:
  PolynomialBoundary(const Polynomials::Polynomial<double> solution_1d)
    : Function<dim>()
    , solution_1d(solution_1d)
  {
  }

  virtual double value(const Point<dim>& p, const unsigned int component = 0) const override;

private:
  Polynomials::Polynomial<double> solution_1d;
};

template <int dim>
class PolynomialError : public AmandusIntegrator<dim>
{
public:
  PolynomialError(const Polynomials::Polynomial<double> solution_1d);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  Polynomials::Polynomial<double> solution_1d;
};

//----------------------------------------------------------------------//

template <int dim>
PolynomialRHS<dim>::PolynomialRHS(const Polynomials::Polynomial<double> solution_1d)
  : solution_1d(solution_1d)
{
  this->use_boundary = false;
  this->use_face = false;
}

template <int dim>
void
PolynomialRHS<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);

  std::vector<double> px(3);
  std::vector<double> py(3);
  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
  {
    const double x = info.fe_values(0).quadrature_point(k)(0);
    const double y = info.fe_values(0).quadrature_point(k)(1);
    solution_1d.value(x, px);
    solution_1d.value(y, py);

    rhs[k] = -px[2] * py[0] - px[0] * py[2];
  }

  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
}

template <int dim>
void
PolynomialRHS<dim>::boundary(DoFInfo<dim>&, IntegrationInfo<dim>&) const
{
}

template <int dim>
void
PolynomialRHS<dim>::face(DoFInfo<dim>&, DoFInfo<dim>&, IntegrationInfo<dim>&,
                         IntegrationInfo<dim>&) const
{
}

//----------------------------------------------------------------------//

template <int dim>
PolynomialResidual<dim>::PolynomialResidual(const Polynomials::Polynomial<double> solution_1d)
  : solution_1d(solution_1d)
{
  this->use_boundary = true;
  this->use_face = true;
}

template <int dim>
void
PolynomialResidual<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));

  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);

  std::vector<double> px(3);
  std::vector<double> py(3);
  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
  {
    const double x = info.fe_values(0).quadrature_point(k)(0);
    const double y = info.fe_values(0).quadrature_point(k)(1);
    solution_1d.value(x, px);
    solution_1d.value(y, py);

    rhs[k] = -px[2] * py[0] - px[0] * py[2];
  }

  double factor = 1.;
  if (this->timestep != 0)
  {
    factor = -this->timestep;
    L2::L2(dinfo.vector(0).block(0), info.fe_values(0), info.values[0][0]);
  }
  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs, -factor);
  Laplace::cell_residual(dinfo.vector(0).block(0), info.fe_values(0), info.gradients[0][0], factor);
}

template <int dim>
void
PolynomialResidual<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  std::vector<double> boundary_values(info.fe_values(0).n_quadrature_points, 0.);
  std::vector<double> px(1);
  std::vector<double> py(1);
  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
  {
    const double x = info.fe_values(0).quadrature_point(k)(0);
    const double y = info.fe_values(0).quadrature_point(k)(1);
    solution_1d.value(x, px);
    solution_1d.value(y, py);
    boundary_values[k] = px[0] * py[0];
  }

  const double factor = (this->timestep == 0.) ? 1. : this->timestep;
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_residual(dinfo.vector(0).block(0),
                            info.fe_values(0),
                            info.values[0][0],
                            info.gradients[0][0],
                            boundary_values,
                            Laplace::compute_penalty(dinfo, dinfo, deg, deg),
                            factor);
}

template <int dim>
void
PolynomialResidual<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2,
                              IntegrationInfo<dim>& info1, IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  const double factor = (this->timestep == 0.) ? 1. : this->timestep;
  Laplace::ip_residual(dinfo1.vector(0).block(0),
                       dinfo2.vector(0).block(0),
                       info1.fe_values(0),
                       info2.fe_values(0),
                       info1.values[0][0],
                       info1.gradients[0][0],
                       info2.values[0][0],
                       info2.gradients[0][0],
                       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
                       factor);
}

//----------------------------------------------------------------------//

template <int dim>
double
PolynomialBoundary<dim>::value(const Point<dim>& p, const unsigned int component) const
{
  Assert(component == 0, ExcNotImplemented());

  std::vector<double> px(1);
  std::vector<double> py(1);
  solution_1d.value(p[0], px);
  solution_1d.value(p[1], py);

  return px[0] * py[0];
}

//----------------------------------------------------------------------//

template <int dim>
PolynomialError<dim>::PolynomialError(const Polynomials::Polynomial<double> solution_1d)
  : solution_1d(solution_1d)
{
  this->use_boundary = false;
  this->use_face = false;
  this->error_types.push_back(2);
  this->error_types.push_back(2);
  this->error_types.push_back(0);
}

template <int dim>
void
PolynomialError<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(dinfo.n_values() >= 2, ExcDimensionMismatch(dinfo.n_values(), 4));
  dinfo.value(2) = 0.;

  std::vector<double> px(3);
  std::vector<double> py(3);
  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
  {
    const double x = info.fe_values(0).quadrature_point(k)(0);
    const double y = info.fe_values(0).quadrature_point(k)(1);
    solution_1d.value(x, px);
    solution_1d.value(y, py);
    const double dx = info.fe_values(0).JxW(k);

    Tensor<1, dim> Du = info.gradients[0][0][k];
    Du[0] -= px[1] * py[0];
    Du[1] -= px[0] * py[1];
    double u = info.values[0][0][k];
    u -= px[0] * py[0];

    // 0. L^2(u)
    dinfo.value(0) += (u * u) * dx;
    // 1. H^1(u)
    dinfo.value(1) += (Du * Du) * dx;
    // 2. Linfty(u)
    if (dinfo.value(2) < u * u)
      dinfo.value(2) = u * u;
  }

  dinfo.value(0) = std::sqrt(dinfo.value(0));
  dinfo.value(1) = std::sqrt(dinfo.value(1));
  dinfo.value(2) = std::sqrt(dinfo.value(2));
}

template <int dim>
void
PolynomialError<dim>::boundary(DoFInfo<dim>&, IntegrationInfo<dim>&) const
{
}

template <int dim>
void
PolynomialError<dim>::face(DoFInfo<dim>&, DoFInfo<dim>&, IntegrationInfo<dim>&,
                           IntegrationInfo<dim>&) const
{
}
}

#endif
