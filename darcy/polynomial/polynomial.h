/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __darcy_polynomial_h
#define __darcy_polynomial_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>
#include <integrator.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

/**
 * Namespace containing local integrator classes to integrate the
 * right hand side, residual and error for a Darcy problem, where the
 * exact solution is a known polynomial.
 *
 * The equations solved are the Darcy equations (with all
 * coefficients equal to one)
 *
 * \f{align*}{
 * u - \nabla p &= f \\
 * \nabla\cdot u &= g
 * \f}
 *
 * The solutions in this namespace are parameterized by three tensor
 * product polynomials \f$\psi\f$, \f$\phi\f$ and \f$\pi\f$ which all have the
 * form
 * \f[
 * p(x,y) = p_{1d}(x)*p_{1d}(y).
 * \f]
 * Given these three polynomials,
 * the solutions are given by the equations
 *
 * \f{align*}{
 * u &= \nabla \times \psi + \nabla \phi \\
 * p &= \phi - \pi
 * \f}
 *
 * The corresponding right hand side is given by
 *
 * \f{align*}{
 * f &= \nabla \times \psi + \nabla \phi \\
 * g &= \Delta \phi
 * \f}
 *
 * @ingroup integrators
 */
namespace Darcy
{
  namespace Polynomial
  {

    template <int dim>
      class RHS : public AmandusIntegrator<dim>
    {
      public:
        RHS(const Polynomials::Polynomial<double> curl_potential_1d,
            const Polynomials::Polynomial<double> grad_potential_1d,
            const Polynomials::Polynomial<double> pressure_1d);

        virtual void cell(DoFInfo<dim>& dinfo,
                          IntegrationInfo<dim>& info) const;
        virtual void boundary(DoFInfo<dim>& dinfo,
                              IntegrationInfo<dim>& info) const;
        virtual void face(DoFInfo<dim>& dinfo1,
                          DoFInfo<dim>& dinfo2,
                          IntegrationInfo<dim>& info1,
                          IntegrationInfo<dim>& info2) const;
      private:
        Polynomials::Polynomial<double> curl_potential_1d;
        Polynomials::Polynomial<double> grad_potential_1d;
        Polynomials::Polynomial<double> pressure_1d;
    };


    /**
     * Integrate the residual for a Darcy problem, where the
     * solution is the curl of the symmetric tensor product of a given
     * polynomial, plus the gradient of another.
     *
     * @ingroup integrators
     */
    template <int dim>
      class Residual : public AmandusIntegrator<dim>
    {
      public:
        Residual(const Polynomials::Polynomial<double> curl_potential_1d,
                 const Polynomials::Polynomial<double> grad_potential_1d,
                 const Polynomials::Polynomial<double> pressure_1d);

        virtual void cell(DoFInfo<dim>& dinfo,
                          IntegrationInfo<dim>& info) const;
        virtual void boundary(DoFInfo<dim>& dinfo,
                              IntegrationInfo<dim>& info) const;
        virtual void face(DoFInfo<dim>& dinfo1,
                          DoFInfo<dim>& dinfo2,
                          IntegrationInfo<dim>& info1,
                          IntegrationInfo<dim>& info2) const;
      private:
        Polynomials::Polynomial<double> curl_potential_1d;
        Polynomials::Polynomial<double> grad_potential_1d;
        Polynomials::Polynomial<double> pressure_1d;
    };


    /**
     * Compute the error with respect to the given solution
     *
     * @ingroup integrators
     */
    template <int dim>
      class Error : public AmandusIntegrator<dim>
    {
      public:
        Error(const Polynomials::Polynomial<double> curl_potential_1d,
              const Polynomials::Polynomial<double> grad_potential_1d,
              const Polynomials::Polynomial<double> pressure_1d);

        virtual void cell(DoFInfo<dim>& dinfo,
                          IntegrationInfo<dim>& info) const;
        virtual void boundary(DoFInfo<dim>& dinfo,
                              IntegrationInfo<dim>& info) const;
        virtual void face(DoFInfo<dim>& dinfo1,
                          DoFInfo<dim>& dinfo2,
                          IntegrationInfo<dim>& info1,
                          IntegrationInfo<dim>& info2) const;
      private:
        Polynomials::Polynomial<double> curl_potential_1d;
        Polynomials::Polynomial<double> grad_potential_1d;
        Polynomials::Polynomial<double> pressure_1d;
    };

    //----------------------------------------------------------------------//

    template <int dim>
      RHS<dim>::RHS(
          const Polynomials::Polynomial<double> curl_potential_1d,
          const Polynomials::Polynomial<double> grad_potential_1d,
          const Polynomials::Polynomial<double> pressure_1d)
      :
        curl_potential_1d(curl_potential_1d),
        grad_potential_1d(grad_potential_1d),
        pressure_1d(pressure_1d)
    {
      this->use_boundary = false;
      this->use_face = false;
    }


    template <int dim>
      void RHS<dim>::cell(
          DoFInfo<dim>& dinfo,
          IntegrationInfo<dim>& info) const
      {
        std::vector<std::vector<double> > rhs_u(
            dim, std::vector<double>(info.fe_values(0).n_quadrature_points));
        std::vector<double> rhs_p(info.fe_values(0).n_quadrature_points);

        std::vector<double> px(2);
        std::vector<double> py(2);
        for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
        {
          const double x = info.fe_values(0).quadrature_point(k)(0);
          const double y = info.fe_values(0).quadrature_point(k)(1);
          curl_potential_1d.value(x, px);
          curl_potential_1d.value(y, py);

          // Right hand side corresponding to the vector potential of the velocity
          rhs_u[0][k] = px[0]*py[1];
          rhs_u[1][k] =-px[1]*py[0];

          // Add a gradient part to the right hand side to test for
          // pressure
          pressure_1d.value(x, px);
          pressure_1d.value(y, py);
          rhs_u[0][k] += px[1]*py[0];
          rhs_u[1][k] += px[0]*py[1];

          // Right hand side corresponding to the scalar potential of the
          // velocity, entering as inhomogeneity for the divergence.

          px.resize(3);
          py.resize(3);
          grad_potential_1d.value(x, px);
          grad_potential_1d.value(y, py);
          rhs_p[k] += px[2]*py[0]+px[0]*py[2];
        }

        L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs_u);
        L2::L2(dinfo.vector(0).block(1), info.fe_values(1), rhs_p);
      }


    template <int dim>
      void RHS<dim>::boundary(
          DoFInfo<dim>&,
          IntegrationInfo<dim>&) const
      {}


    template <int dim>
      void RHS<dim>::face(
          DoFInfo<dim>&,
          DoFInfo<dim>&,
          IntegrationInfo<dim>&,
          IntegrationInfo<dim>&) const
      {}

    //----------------------------------------------------------------------//

    template <int dim>
      Residual<dim>::Residual(
          const Polynomials::Polynomial<double> curl_potential_1d,
          const Polynomials::Polynomial<double> grad_potential_1d,
          const Polynomials::Polynomial<double> pressure_1d)
      :
        curl_potential_1d(curl_potential_1d),
        grad_potential_1d(grad_potential_1d),
        pressure_1d(pressure_1d)
    {
      this->use_boundary = false;
      this->use_face = false;
      //this->input_vector_names.push_back("Newton iterate");
    }


    template <int dim>
      void Residual<dim>::cell(
          DoFInfo<dim>& dinfo,
          IntegrationInfo<dim>& info) const
      {
        Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
        Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));

        std::vector<std::vector<double> > rhs_u(
            dim, std::vector<double>(info.fe_values(0).n_quadrature_points));
        std::vector<double> rhs_p(info.fe_values(0).n_quadrature_points);

        std::vector<double> px(2);
        std::vector<double> py(2);
        for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
        {
          const double x = info.fe_values(0).quadrature_point(k)(0);
          const double y = info.fe_values(0).quadrature_point(k)(1);
          curl_potential_1d.value(x, px);
          curl_potential_1d.value(y, py);

          // Right hand side corresponding to the vector potential of the velocity
          rhs_u[0][k] = px[0]*py[1];
          rhs_u[1][k] =-px[1]*py[0];

          // Add a gradient part to the right hand side to test for
          // pressure
          pressure_1d.value(x, px);
          pressure_1d.value(y, py);
          rhs_u[0][k] += px[1]*py[0];
          rhs_u[1][k] += px[0]*py[1];

          // Right hand side corresponding to the scalar potential of the
          // velocity, entering as inhomogeneity for the divergence.

          px.resize(3);
          py.resize(3);
          grad_potential_1d.value(x, px);
          grad_potential_1d.value(y, py);
          rhs_p[k] += px[2]*py[0]+px[0]*py[2];
        }

        L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs_u, -1.);
        L2::L2(dinfo.vector(0).block(1), info.fe_values(1), rhs_p, -1.);

        L2::L2(dinfo.vector(0).block(0), info.fe_values(0),
               make_slice(info.values[0], 0, dim));
        Divergence::gradient_residual(dinfo.vector(0).block(0), info.fe_values(0),
                                      info.values[0][dim], -1.);
        Divergence::cell_residual(dinfo.vector(0).block(1), info.fe_values(1),
                                  make_slice(info.gradients[0], 0, dim), 1.);
      }


    template <int dim>
      void Residual<dim>::boundary(
          DoFInfo<dim>&,
          IntegrationInfo<dim>&) const
      {}


    template <int dim>
      void Residual<dim>::face(
          DoFInfo<dim>&,
          DoFInfo<dim>&,
          IntegrationInfo<dim>&,
          IntegrationInfo<dim>&) const
      {}

    //----------------------------------------------------------------------//

    template <int dim>
      Error<dim>::Error(
          const Polynomials::Polynomial<double> curl_potential_1d,
          const Polynomials::Polynomial<double> grad_potential_1d,
          const Polynomials::Polynomial<double> pressure_1d)
      :
        curl_potential_1d(curl_potential_1d),
        grad_potential_1d(grad_potential_1d),
        pressure_1d(pressure_1d)
    {
      this->num_errors = 5;
      this->use_boundary = false;
      this->use_face = false;
    }


    template <int dim>
      void Error<dim>::cell(
          DoFInfo<dim>& dinfo,
          IntegrationInfo<dim>& info) const
      {
        std::vector<double> px(3);
        std::vector<double> py(3);
        for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
        {
          const double x = info.fe_values(0).quadrature_point(k)(0);
          const double y = info.fe_values(0).quadrature_point(k)(1);
          curl_potential_1d.value(x, px);
          curl_potential_1d.value(y, py);
          const double dx = info.fe_values(0).JxW(k);

          Tensor<1,dim> Du0 = info.gradients[0][0][k];
          Du0[0] -= px[1]*py[1];
          Du0[1] -= px[0]*py[2];
          Tensor<1,dim> Du1 = info.gradients[0][1][k];
          Du1[0] += px[2]*py[0];
          Du1[1] += px[1]*py[1];
          double u0 = info.values[0][0][k];
          u0 -= px[0]*py[1];
          double u1 = info.values[0][1][k];
          u1 += px[1]*py[0];

          pressure_1d.value(x, px);
          pressure_1d.value(y, py);
          double p = info.values[0][dim][k];
          p += px[0]*py[0];
          Tensor<1,dim> Dp = info.gradients[0][dim][k];
          Dp[0] += px[1]*py[0];
          Dp[1] += px[0]*py[1];

          grad_potential_1d.value(x, px);
          grad_potential_1d.value(y, py);
          u0 -= px[1]*py[0];
          u1 -= px[0]*py[1];
          Du0[0] -= px[2]*py[0];
          Du0[1] -= px[1]*py[1];
          Du1[0] -= px[1]*py[1];
          Du1[1] -= px[0]*py[2];

          double divu = Du0[0] + Du1[1];
          p -= px[0]*py[0];
          Dp[0] -= px[1]*py[0];
          Dp[1] -= px[0]*py[1];

          // 0. L^2(u)
          dinfo.value(0) += (u0*u0+u1*u1) * dx;
          // 1. H^1(u)
          dinfo.value(1) += ((Du0*Du0)+(Du1*Du1)) * dx;
          // 2. div u
          dinfo.value(2) += divu*divu * dx;
          // 3. L^2(p) up to mean value
          dinfo.value(3) += p*p * dx;
          // 4. H^1(p)
          dinfo.value(4) += Dp*Dp * dx;
        }

        for (unsigned int i=0;i<=4;++i) {
          dinfo.value(i) = std::sqrt(dinfo.value(i));
        }
      }


    template <int dim>
      void Error<dim>::boundary(
          DoFInfo<dim>&,
          IntegrationInfo<dim>&) const
      {}


    template <int dim>
      void Error<dim>::face(
          DoFInfo<dim>&,
          DoFInfo<dim>&,
          IntegrationInfo<dim>&,
          IntegrationInfo<dim>&) const
      {}

  }
}

#endif
