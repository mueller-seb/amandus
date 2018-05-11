#ifndef biot_polynomial_h
#define biot_polynomial_h

#include <amandus/biot/biot.h>
#include <deal.II/base/polynomial.h>

namespace Biot
{
/**
 * Polynomial solution to stationary, linear Biot.
 */
template <int dim>
class PolynomialResidual : public Residual<dim>
{
public:
  PolynomialResidual(const std::vector<dealii::Polynomials::Polynomial<double>>& polynomials,
                     const Parameters& par, bool implicit = true);
  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;

private:
  /**
   * A vector with  entries.
   *
   * <ol>
   * <li>2 respectively 9 polynomials for the components of the curl potential of the elastic
   * deformation</li>
   * <li>2 respectively 3 polynomials for the gradient potential of elastic deformation</li>
   * <li>2 respectively 9 polynomials for the components of the curl potential of the seepage
   * velocity</li>
   * <li>2 respectively 3 polynomials for the gradient potential of seepage velocity</li>
   * <li> 2 resp 3 tensor factors for the fluid pressure</li>
   * </ol>
   */
  const std::vector<dealii::Polynomials::Polynomial<double>> poly;
};

/**
 * Polynomial solution to stationary, linear Biot.
 */
template <int dim>
class PolynomialError : public AmandusIntegrator<dim>
{
public:
  PolynomialError(const std::vector<dealii::Polynomials::Polynomial<double>>& polynomials,
                  const Parameters& par);
  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;

private:
  dealii::SmartPointer<const Parameters, class Residual<dim>> parameters;
  /**
   * See PolynomialResidual::poly
   */
  const std::vector<dealii::Polynomials::Polynomial<double>> poly;
};

template <int dim>
PolynomialResidual<dim>::PolynomialResidual(
  const std::vector<dealii::Polynomials::Polynomial<double>>& polynomials, const Parameters& par,
  bool implicit)
  : Residual<dim>(par, implicit)
  , poly(polynomials)
{
}

template <int dim>
void
PolynomialResidual<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  // The operator residual for nonlinear iterations
  if (info.values.size() > 0)
    Residual<dim>::cell(dinfo, info);

  // Vectors to be filled with function values
  std::vector<std::vector<double>> rhs_u(
    dim, std::vector<double>(info.fe_values(0).n_quadrature_points));
  std::vector<std::vector<double>> rhs_w(
    dim, std::vector<double>(info.fe_values(0).n_quadrature_points));
  std::vector<std::vector<double>> rhs_p(
    1, std::vector<double>(info.fe_values(0).n_quadrature_points));

  //  Vector storing polynomial values
  std::vector<std::vector<double>> phi(dim, std::vector<double>(4));

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
  {
    const dealii::Point<dim>& x = info.fe_values(0).quadrature_point(k);
    double Divu = 0.;
    dealii::Tensor<1, dim> DivDu;
    dealii::Tensor<1, dim> DDivu;
    dealii::Tensor<1, dim> w;
    double Divw = 0.;
    dealii::Tensor<1, dim> Gradp;

    // The displacement

    // The curl potential
    poly[0].value(x(0), phi[0]);
    poly[1].value(x(1), phi[1]);
    // div epsilon(curl) = Delta
    // grad div curl = 0
    DivDu[0] += 0.5 * (phi[0][0] * phi[1][3] + phi[0][2] * phi[1][1]);
    DivDu[1] -= 0.5 * (phi[0][3] * phi[1][0] + phi[0][1] * phi[1][2]);
    // The gradient potential
    poly[2].value(x(0), phi[0]);
    poly[3].value(x(1), phi[1]);
    // div u
    Divu -= phi[0][2] * phi[1][0] + phi[0][0] * phi[1][2];
    // div epsilon(u) and grad div u
    DivDu[0] -= phi[0][3] * phi[1][0] + phi[0][1] * phi[1][2];
    DivDu[1] -= phi[0][0] * phi[1][3] + phi[0][2] * phi[1][1];
    DDivu[0] -= phi[0][3] * phi[1][0] + phi[0][1] * phi[1][2];
    DDivu[1] -= phi[0][2] * phi[1][1] + phi[0][0] * phi[1][3];

    // Seepage velocity

    // The curl potential
    poly[4].value(x(0), phi[0]);
    poly[5].value(x(1), phi[1]);
    w[0] += phi[0][0] * phi[1][1];
    w[1] -= phi[0][1] * phi[1][0];
    // The gradient potential
    poly[6].value(x(0), phi[0]);
    poly[7].value(x(1), phi[1]);
    // div w
    Divw -= phi[0][2] * phi[1][0] + phi[0][0] * phi[1][2];
    w[0] -= phi[0][1] * phi[1][0];
    w[1] -= phi[0][0] * phi[1][1];

    // Pressure
    poly[8].value(x(0), phi[0]);
    poly[9].value(x(1), phi[1]);
    for (unsigned int d1 = 0; d1 < dim; ++d1)
    {
      Gradp[d1] = -1.;
      for (unsigned int d2 = 0; d2 < dim; ++d2)
        Gradp[d1] *= phi[d2][(d1 == d2 ? 1 : 0)];
    }
    for (unsigned int d = 0; d < dim; ++d)
    {
      rhs_u[d][k] = 2. * this->parameters->mu * DivDu[d] + this->parameters->lambda * DDivu[d] -
                    this->parameters->p_to_disp * Gradp[d];
      rhs_w[d][k] = -this->parameters->resistance * w[d] - Gradp[d];
    }
    rhs_p[0][k] = Divw;
  }
  // We have to multiply by -1 if this is inside a Newton method.
  const double factor =
    (info.values.size() > 0) ? (this->timestep == 0. ? -1. : -this->timestep) : 1.;
  dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs_u, factor);
  dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(1), info.fe_values(0), rhs_w, factor);
  dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(2), info.fe_values(1), rhs_p, factor);
}

template <int dim>
void
PolynomialResidual<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  if (info.values.size() == 0)
    return;
  const double factor =
    (this->timestep == 0.) ? 1. : (this->is_implicit ? this->timestep : -this->timestep);
  const double mu = factor * this->parameters->mu;

  std::vector<std::vector<double>> null(
    2 * dim + 1, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));

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
PolynomialError<dim>::PolynomialError(
  const std::vector<dealii::Polynomials::Polynomial<double>>& polynomials, const Parameters& par)
  : parameters(&par)
  , poly(polynomials)
{
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
  this->use_boundary = false;
  this->use_face = false;
}

template <int dim>
void
PolynomialError<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  //  Vector storing polynomial values
  std::vector<std::vector<double>> phi(dim, std::vector<double>(4));

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
  {
    const dealii::Point<dim>& x = info.fe_values(0).quadrature_point(k);
    //double Divu = 0.;
    dealii::Tensor<1, dim> u;
    dealii::Tensor<2, dim> Du;
    dealii::Tensor<1, dim> w;
    dealii::Tensor<2, dim> Dw;
    //double Divw = 0.;
    double p;
    dealii::Tensor<1, dim> Dp;

    for (unsigned int d = 0; d < dim; ++d)
    {
      u[d] = info.values[0][d][k];
      Du[d] = info.gradients[0][d][k];
      w[d] = info.values[0][d + dim][k];
      Dw[d] = info.gradients[0][d + dim][k];
    }
    p = info.values[0][2 * dim][k];
    Dp = info.gradients[0][2 * dim][k];

    // The displacement

    // The curl potential
    poly[0].value(x(0), phi[0]);
    poly[1].value(x(1), phi[1]);
    u[0] += phi[0][0] * phi[1][1];
    u[1] -= phi[0][1] * phi[1][0];
    Du[0][0] += phi[0][1] * phi[1][1];
    Du[0][1] += phi[0][0] * phi[1][2];
    Du[1][0] -= phi[0][2] * phi[1][0];
    Du[1][1] -= phi[0][1] * phi[1][1];
    // The gradient potential
    poly[2].value(x(0), phi[0]);
    poly[3].value(x(1), phi[1]);
    u[0] -= phi[0][1] * phi[1][0];
    u[1] -= phi[0][0] * phi[1][1];
    Du[0][0] -= phi[0][2] * phi[1][0];
    Du[0][1] -= phi[0][1] * phi[1][1];
    Du[1][0] -= phi[0][1] * phi[1][1];
    Du[1][1] -= phi[0][0] * phi[1][2];

    // Seepage velocity

    // The curl potential
    poly[4].value(x(0), phi[0]);
    poly[5].value(x(1), phi[1]);
    w[0] += phi[0][0] * phi[1][1];
    w[1] -= phi[0][1] * phi[1][0];
    Dw[0][0] += phi[0][1] * phi[1][1];
    Dw[0][1] += phi[0][0] * phi[1][2];
    Dw[1][0] -= phi[0][2] * phi[1][0];
    Dw[1][1] -= phi[0][1] * phi[1][1];
    // The gradient potential
    poly[6].value(x(0), phi[0]);
    poly[7].value(x(1), phi[1]);
    w[0] -= phi[0][1] * phi[1][0];
    w[1] -= phi[0][0] * phi[1][1];
    Dw[0][0] -= phi[0][2] * phi[1][0];
    Dw[0][1] -= phi[0][1] * phi[1][1];
    Dw[1][0] -= phi[0][1] * phi[1][1];
    Dw[1][1] -= phi[0][0] * phi[1][2];

    // Pressure
    poly[8].value(x(0), phi[0]);
    poly[9].value(x(1), phi[1]);
    p += phi[0][0] * phi[1][0];
    for (unsigned int d1 = 0; d1 < dim; ++d1)
    {
      double s = 1.;
      for (unsigned int d2 = 0; d2 < dim; ++d2)
        s *= phi[d2][(d1 == d2 ? 1 : 0)];
      Dp[d1] += s;
    }

    double uu = 0., Duu = 0., divuu = 0., ww = 0., Dww = 0., divww = 0., pp = 0., Dpp = 0.;
    for (unsigned int d = 0; d < dim; ++d)
    {
      uu += u[d] * u[d];
      Duu += Du[d] * Du[d];
      divuu += Du[d][d] * Du[d][d];
      ww += w[d] * w[d];
      Dww += Dw[d] * Dw[d];
      divww += Dw[d][d] * Dw[d][d];
      Dpp += Dp[d] * Dp[d];
    }
    pp = p * p;

    const double dx = info.fe_values(0).JxW(k);

    // 0. L^2(u)
    dinfo.value(0) += uu * dx;
    // 1. H^1(u)
    dinfo.value(1) += Duu * dx;
    // 2. div u
    dinfo.value(2) += divuu * dx;
    // 3. L^2(u)
    dinfo.value(3) += ww * dx;
    // 4. H^1(u)
    dinfo.value(4) += Dww * dx;
    // 5. div u
    dinfo.value(5) += divww * dx;
    dinfo.value(6) += pp * dx;
    dinfo.value(7) += Dpp * dx;
  }
  for (unsigned int i = 0; i < dinfo.n_values(); ++i)
    dinfo.value(i) = std::sqrt(dinfo.value(i));
}
}

#endif
