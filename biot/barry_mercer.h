#ifndef barry_mercer__h
#define barry_mercer__h

#include <algorithm>
#include <array>
#include <cmath>
#include <deal.II/base/point.h>
#include <deal.II/integrators/elasticity.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

#include <amandus/biot/biot.h>

/**
 * The Fourier coefficients of a Dirac functional.
 */
template <int dim>
class CoefficientsDirac
{
  dealii::Point<dim> location;
  const unsigned int nn = 40;

public:
  CoefficientsDirac()
  {
    location(0) = .25;
    location(1) = .25;
    if (dim > 2)
      location(2) = .25;
  }

  /**
   * The number of coefficients in each coordinate direction.
   */
  unsigned int
  n(unsigned int) const
  {
    return nn;
  }

  double
  Qbar(unsigned int n, unsigned int q) const
  {
    return 2. * std::sin(M_PI * n * location(0)) * std::sin(M_PI * q * location(1));
  }
};

/**
 * The Fourier coefficients of a smooth right hand side.
 */
template <int dim>
class Coefficients1Sine
{
  const unsigned int nn = 2;

public:
  /**
   * The number of coefficients in each coordinate direction.
   */
  unsigned int
  n(unsigned int) const
  {
    return nn;
  }

  double
  Qbar(unsigned int n, unsigned int q) const
  {
    return (n == 1 && q == 1) ? 1. : 0.;
  }
};

/**
 * The Fourier coefficients of a smooth right hand side.
 */
template <int dim>
class Coefficients2Sine
{
  const unsigned int nn = 3;

public:
  /**
   * The number of coefficients in each coordinate direction.
   */
  unsigned int
  n(unsigned int) const
  {
    return nn;
  }

  double
  Qbar(unsigned int n, unsigned int q) const
  {
    return (n == 2 && q == 2) ? .25 : 0.;
  }
};

/**
 * The Fourier coefficients of a square wave in x and y
 */
template <int dim>
class CoefficientsRect
{
  std::array<unsigned int, dim> nn;

public:
  CoefficientsRect(unsigned int n)
  {
    for (unsigned int d = 0; d < dim; ++d)
      nn[d] = n;
  }
  /**
   * The number of coefficients in each coordinate direction.
   */
  unsigned int
  n(unsigned int i) const
  {
    return nn[i];
  }

  double
  Qbar(unsigned int n, unsigned int q) const
  {
    if (n % 2 == 0 && q % 2 == 0)
      if ((n / 2) % 2 != 0 && (q / 2) % 2 != 0)
        return 16. / M_PI / M_PI * 1. / n * 1. / q;
      else
        return 0.;
    else
      return 0.;
  }
};

/**
 * Compute solution of Biot's equations for a right hand side
 * oscillating in time.
 */
template <int dim>
class BarryMercer
{
  const double omega = 2. * M_PI;
  // $1+\lambda/\mu$
  const double m;

public:
  BarryMercer(const double mu, const double lambda)
    : m(1. + lambda / mu)
  {
  }

  template <class C>
  std::array<double, 2 * dim + 1>
  operator()(double t, const dealii::Point<dim>& x, const C& coeff) const
  {
    std::array<double, 2 * dim + 1> result;
    std::fill(result.begin(), result.end(), 0.);
    for (unsigned i = 1; i < coeff.n(0); ++i)
      for (unsigned j = 1; j < coeff.n(1); ++j)
      {
        const double co = coeff.Qbar(i, j);
        if (co == 0.)
          continue;
        const double li = M_PI * i;
        const double lj = M_PI * j;
        const double lhat = li * li + lj * lj;
        const double g1 =
          lhat * std::sin(omega * t) - omega * std::cos(omega * t) + omega * std::exp(-lhat * t);

        const double tmp = g1 / (lhat * lhat + omega * omega) * co;
        result[0] += 4. / (m + 1.) * tmp * li / lhat * std::cos(li * x(0)) * std::sin(lj * x(1));
        result[1] += 4. / (m + 1.) * tmp * lj / lhat * std::sin(li * x(0)) * std::cos(lj * x(1));
        result[dim] += 4. * tmp * li * std::cos(li * x(0)) * std::sin(lj * x(1));
        result[dim + 1] += 4. * tmp * lj * std::sin(li * x(0)) * std::cos(lj * x(1));
        result[2 * dim] += 4. * tmp * std::sin(li * x(0)) * std::sin(lj * x(1));
      }
    return result;
  }

  template <class C>
  std::array<dealii::Tensor<1, dim>, 2 * dim + 1>
  grad(double t, const dealii::Point<dim>& x, const C& coeff) const
  {
    std::array<dealii::Tensor<1, dim>, 2 * dim + 1> result;
    std::fill(result.begin(), result.end(), 0.);
    for (unsigned i = 1; i < coeff.n(0); ++i)
      for (unsigned j = 1; j < coeff.n(1); ++j)
      {
        const double co = coeff.Qbar(i, j);
        if (co == 0.)
          continue;
        const double li = M_PI * i;
        const double lj = M_PI * j;
        const double lhat = li * li + lj * lj;
        const double g1 =
          lhat * std::sin(omega * t) - omega * std::cos(omega * t) + omega * std::exp(-lhat * t);

        const double tmp = g1 / (lhat * lhat + omega * omega) * co;
        result[0][0] -=
          4. / (m + 1.) * tmp * li * li / lhat * std::sin(li * x(0)) * std::sin(lj * x(1));
        result[0][1] +=
          4. / (m + 1.) * tmp * li * lj / lhat * std::cos(li * x(0)) * std::cos(lj * x(1));
        result[1][0] +=
          4. / (m + 1.) * tmp * lj * li / lhat * std::cos(li * x(0)) * std::cos(lj * x(1));
        result[1][1] -=
          4. / (m + 1.) * tmp * lj * lj / lhat * std::sin(li * x(0)) * std::sin(lj * x(1));
        result[dim][0] -= 4. * tmp * li * li * std::sin(li * x(0)) * std::sin(lj * x(1));
        result[dim][1] += 4. * tmp * li * lj * std::cos(li * x(0)) * std::cos(lj * x(1));
        result[dim + 1][0] += 4. * tmp * lj * li * std::cos(li * x(0)) * std::cos(lj * x(1));
        result[dim + 1][1] -= 4. * tmp * lj * lj * std::sin(li * x(0)) * std::sin(lj * x(1));
        result[2 * dim][0] += 4. * tmp * li * std::cos(li * x(0)) * std::sin(lj * x(1));
        result[2 * dim][1] += 4. * tmp * lj * std::sin(li * x(0)) * std::cos(lj * x(1));
      }
    return result;
  }
};

namespace Biot
{
template <int dim, class COEFF = Coefficients2Sine<dim>>
class BarryMercerError : public AmandusIntegrator<dim>
{
  dealii::SmartPointer<const Parameters, class BarryMercerError<dim, COEFF>> parameters;
  double time;
  COEFF coeff;

public:
  BarryMercerError(const Parameters& par, double t, const COEFF& c = Coefficients2Sine<dim>())
    : parameters(&par)
    , time(t)
    , coeff(c)
  {
    this->use_boundary = false;
    this->use_face = false;
    this->error_types.push_back(2);
    this->error_names.push_back("L2u");
    this->error_types.push_back(2);
    this->error_names.push_back("H1u");
    this->error_types.push_back(2);
    this->error_names.push_back("L2divu");
    this->error_types.push_back(2);
    this->error_names.push_back("L2w");
    this->error_types.push_back(2);
    this->error_names.push_back("H1w");
    this->error_types.push_back(2);
    this->error_names.push_back("L2divw");
    this->error_types.push_back(2);
    this->error_names.push_back("L2p");
    this->error_types.push_back(2);
    this->error_names.push_back("H1p");
  }

  virtual void
  cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
  {
    BarryMercer<dim> solution(parameters->mu, parameters->lambda);
    const unsigned int nq = info.fe_values(0).n_quadrature_points;
    const unsigned int sqrt_nq = static_cast<unsigned int>(std::sqrt(nq) + .5);
    // std::cout << std::endl;
    for (unsigned int k = 0; k < nq; ++k)
    {
      const dealii::Point<dim>& x = info.fe_values(0).quadrature_point(k);
      auto exact = solution(time, x, coeff);
      auto grad = solution.grad(time, x, coeff);
      dealii::Tensor<1, dim> eu;
      dealii::Tensor<2, dim> Du;
      dealii::Tensor<1, dim> ew;
      dealii::Tensor<2, dim> Dw;
      double ep;
      dealii::Tensor<1, dim> Dp;

      // The else branch is here so that we can compute norms of
      // the exact solution
      if (true)
      {
        for (unsigned int d = 0; d < dim; ++d)
        {
          eu[d] = info.values[0][d][k] - exact[d];
          Du[d] = info.gradients[0][d][k] - grad[d];
          ew[d] = info.values[0][d + dim][k] - exact[d + dim];
          Dw[d] = info.gradients[0][d + dim][k] - grad[d + dim];
        }
        ep = info.values[0][2 * dim][k] - exact[2 * dim];
        Dp = info.gradients[0][2 * dim][k] - grad[2 * dim];
      }
      else
      {
        for (unsigned int d = 0; d < dim; ++d)
        {
          eu[d] = exact[d];
          Du[d] = grad[d];
          ew[d] = exact[d + dim];
          Dw[d] = grad[d + dim];
        }
        ep = exact[2 * dim];
        Dp = grad[2 * dim];
      }

      double uu = 0., Duu = 0., divu = 0., ww = 0., Dww = 0., divw = 0., pp = 0., Dpp = 0.;
      for (unsigned int d = 0; d < dim; ++d)
      {
        uu += eu[d] * eu[d];
        Duu += Du[d] * Du[d];
        divu += Du[d][d];
        ww += ew[d] * ew[d];
        Dww += Dw[d] * Dw[d];
        divw += Dw[d][d];
        Dpp += Dp[d] * Dp[d];
      }
      pp = ep * ep;

      const double dx = info.fe_values(0).JxW(k);

      // 0. L^2(u)
      dinfo.value(0) += uu * dx;
      // 1. H^1(u)
      dinfo.value(1) += Duu * dx;
      // 2. div u
      dinfo.value(2) += divu * divu * dx;
      // 3. L^2(w)
      dinfo.value(3) += ww * dx;
      // 4. H^1(w)
      dinfo.value(4) += Dww * dx;
      // 5. div w
      dinfo.value(5) += divw * divw * dx;
      dinfo.value(6) += pp * dx;
      dinfo.value(7) += Dpp * dx;
    }
    for (unsigned int i = 0; i < dinfo.n_values(); ++i)
      dinfo.value(i) = std::sqrt(dinfo.value(i));
  }
};

template <int dim>
class BarryMercerMatrix : public Matrix<dim>
{
public:
  BarryMercerMatrix(const Parameters& par)
    : Matrix<dim>(par)
  {
  }

  virtual void
  boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
  {
    const double factor = (this->timestep == 0.) ? 1. : this->timestep;
    const double mu = factor * this->parameters->mu;

    const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
    // Elasticity
    dealii::LocalIntegrators::Elasticity::nitsche_tangential_matrix(
      dinfo.matrix(0, false).matrix,
      info.fe_values(0),
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * mu);
  }
};

template <int dim>
class BarryMercerSource : public Residual<dim>
{
public:
  BarryMercerSource(const Parameters& par, bool implicit = true)
    : Residual<dim>(par, implicit)
  {
  }

  virtual void
  boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
  {
    const double factor =
      (this->timestepping) ? (this->is_implicit ? this->timestep : -this->timestep) : 1.;
    const double mu = factor * this->parameters->mu;
    std::vector<std::vector<double>> null = std::vector<std::vector<double>>(
      dim, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));

    const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
    // Elasticity
    dealii::LocalIntegrators::Elasticity::nitsche_tangential_residual(
      dinfo.vector(0).block(0),
      info.fe_values(0),
      dealii::make_slice(info.values[0], 0, dim),
      dealii::make_slice(info.gradients[0], 0, dim),
      null,
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * mu);
  }

  virtual void
  cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
  {
    const double factor =
      (this->timestepping) ? (this->is_implicit ? this->timestep : -this->timestep) : 1.;
    std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);
    for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
      rhs[k] = -(2 * this->parameters->mu + this->parameters->lambda) *
               std::sin(2 * M_PI * info.fe_values(0).quadrature_point(k)(0)) *
               std::sin(2 * M_PI * info.fe_values(0).quadrature_point(k)(1)) *
               std::sin(2 * M_PI * this->time);
    dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(2), info.fe_values(1), rhs, factor);
    Residual<dim>::cell(dinfo, info);
  }
};

template <int dim>
class BarryMercerRect : public Residual<dim>
{
public:
  BarryMercerRect(const Parameters& par, bool implicit = true)
    : Residual<dim>(par, implicit)
  {
  }

  virtual void
  boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
  {
    const double factor =
      (this->timestepping) ? (this->is_implicit ? this->timestep : -this->timestep) : 1.;
    const double mu = factor * this->parameters->mu;
    std::vector<std::vector<double>> null = std::vector<std::vector<double>>(
      dim, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));

    const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
    // Elasticity
    dealii::LocalIntegrators::Elasticity::nitsche_tangential_residual(
      dinfo.vector(0).block(0),
      info.fe_values(0),
      dealii::make_slice(info.values[0], 0, dim),
      dealii::make_slice(info.gradients[0], 0, dim),
      null,
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * mu);
  }

  virtual void
  cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
  {
    const double factor =
      (this->timestepping) ? (this->is_implicit ? this->timestep : -this->timestep) : 1.;
    std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);
    for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
      rhs[k] = (info.fe_values(0).quadrature_point(k)(0) < .5 ? 1. : -1.) *
               (info.fe_values(0).quadrature_point(k)(1) < .5 ? 1. : -1.) *
               std::sin(2 * M_PI * this->time);
    dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(2), info.fe_values(1), rhs, -factor);
    Residual<dim>::cell(dinfo, info);
  }
};
}

#endif
