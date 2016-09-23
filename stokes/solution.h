/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __stokes_solution_h
#define __stokes_solution_h

#include <amandus/integrator.h>
#include <deal.II/base/flow_function.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;
using namespace Functions;

namespace StokesIntegrators
{
/**
 * Integrate the right hand side for a Stokes problem, where the
 * solution is given by a FlowFunction.
 *
 * @ingroup integrators
 */
template <int dim>
class SolutionRHS : public AmandusIntegrator<dim>
{
public:
  SolutionRHS(FlowFunction<dim>& solution);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const;

private:
  SmartPointer<FlowFunction<dim>, SolutionRHS<dim>> solution;
};

/**
 * Integrate the residual for a Stokes problem, where the
 * solution is given by a FlowFunction.
 *
 * @ingroup integrators
 */
template <int dim>
class SolutionResidual : public AmandusIntegrator<dim>
{
public:
  SolutionResidual(FlowFunction<dim>& solution);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const;

private:
  SmartPointer<FlowFunction<dim>, SolutionResidual<dim>> solution;
};

template <int dim>
class SolutionError : public AmandusIntegrator<dim>
{
public:
  SolutionError(FlowFunction<dim>& solution);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const;

private:
  SmartPointer<FlowFunction<dim>, SolutionError<dim>> solution;
};

//----------------------------------------------------------------------//

template <int dim>
SolutionRHS<dim>::SolutionRHS(FlowFunction<dim>& solution)
  : solution(&solution)
{
  this->use_boundary = true;
  this->use_face = false;
}

template <int dim>
void
SolutionRHS<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  std::vector<std::vector<double>> rhs(
    dim + 1, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));

  // This is not the real laplace but the rhs for stokes
  solution->vector_laplacians(info.fe_values(0).get_quadrature_points(), rhs);

  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs, -1.);
}

template <int dim>
void
SolutionRHS<dim>::boundary(DoFInfo<dim>&, IntegrationInfo<dim>&) const
{
}

template <int dim>
void
SolutionRHS<dim>::face(DoFInfo<dim>&, DoFInfo<dim>&, IntegrationInfo<dim>&,
                       IntegrationInfo<dim>&) const
{
}

//----------------------------------------------------------------------//

template <int dim>
SolutionResidual<dim>::SolutionResidual(FlowFunction<dim>& solution)
  : solution(&solution)
{
  this->use_boundary = true;
  this->use_face = true;
  this->input_vector_names.push_back("Newton iterate");
}

template <int dim>
void
SolutionResidual<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));

  std::vector<std::vector<double>> rhs(
    dim + 1, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));

  // This is not the real laplace but the rhs for stokes
  solution->vector_laplacians(info.fe_values(0).get_quadrature_points(), rhs);

  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), make_slice(rhs, 0, dim));
  Laplace::cell_residual(
    dinfo.vector(0).block(0), info.fe_values(0), make_slice(info.gradients[0], 0, dim));
  Divergence::gradient_residual(
    dinfo.vector(0).block(0), info.fe_values(0), info.values[0][dim], -1.);
  Divergence::cell_residual(
    dinfo.vector(0).block(1), info.fe_values(1), make_slice(info.gradients[0], 0, dim), 1.);
}

template <int dim>
void
SolutionResidual<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  std::vector<std::vector<double>> values(
    dim + 1, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));
  solution->vector_values(info.fe_values(0).get_quadrature_points(), values);

  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_residual(dinfo.vector(0).block(0),
                            info.fe_values(0),
                            make_slice(info.values[0], 0, dim),
                            make_slice(info.gradients[0], 0, dim),
                            make_slice(values, 0, dim),
                            Laplace::compute_penalty(dinfo, dinfo, deg, deg));
}

template <int dim>
void
SolutionResidual<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                            IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Laplace::ip_residual(dinfo1.vector(0).block(0),
                       dinfo2.vector(0).block(0),
                       info1.fe_values(0),
                       info2.fe_values(0),
                       make_slice(info1.values[0], 0, dim),
                       make_slice(info1.gradients[0], 0, dim),
                       make_slice(info2.values[0], 0, dim),
                       make_slice(info2.gradients[0], 0, dim),
                       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
}

//----------------------------------------------------------------------//

template <int dim>
SolutionError<dim>::SolutionError(FlowFunction<dim>& solution)
  : solution(&solution)
{
  this->use_boundary = false;
  this->use_face = false;
  this->num_errors = 5;
}

template <int dim>
void
SolutionError<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(dinfo.n_values() >= 5, ExcDimensionMismatch(dinfo.n_values(), 5));

  std::vector<std::vector<double>> val(
    dim + 1, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));
  std::vector<std::vector<Tensor<1, dim>>> grad(
    dim + 1, std::vector<Tensor<1, dim>>(info.fe_values(0).n_quadrature_points));

  solution->vector_values(info.fe_values(0).get_quadrature_points(), val);
  solution->vector_gradients(info.fe_values(0).get_quadrature_points(), grad);

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
  {
    const double dx = info.fe_values(0).JxW(k);

    Tensor<1, dim> Du0 = info.gradients[0][0][k];
    double div = Du0[0];
    Du0[0] -= grad[0][k][0];
    Du0[1] -= grad[0][k][1];
    Tensor<1, dim> Du1 = info.gradients[0][1][k];
    div += Du1[1];
    Du1[0] -= grad[1][k][0];
    Du1[1] -= grad[1][k][1];
    div -= grad[0][k][0] + grad[1][k][1];
    double u0 = info.values[0][0][k];
    u0 -= val[0][k];
    double u1 = info.values[0][1][k];
    u1 -= val[1][k];

    double p = info.values[0][dim][k];
    p += val[dim][k];
    Tensor<1, dim> Dp = info.gradients[0][dim][k];
    Dp[0] += grad[dim][k][0];
    Dp[1] += grad[dim][k][1];

    // 0. L^2(u)
    dinfo.value(0) += (u0 * u0 + u1 * u1) * dx;
    // 1. H^1(u)
    dinfo.value(1) += ((Du0 * Du0) + (Du1 * Du1)) * dx;
    // 2. div u
    dinfo.value(2) = (div * div) * dx;
    // 3. L^2(p) up to mean value
    dinfo.value(3) = (p * p) * dx;
    // 4. H^1(p)
    dinfo.value(4) = (Dp * Dp) * dx;
  }
  for (unsigned int i = 0; i <= 4; ++i)
    dinfo.value(i) = std::sqrt(dinfo.value(i));
}

template <int dim>
void
SolutionError<dim>::boundary(DoFInfo<dim>&, IntegrationInfo<dim>&) const
{
}

template <int dim>
void
SolutionError<dim>::face(DoFInfo<dim>&, DoFInfo<dim>&, IntegrationInfo<dim>&,
                         IntegrationInfo<dim>&) const
{
}

/*
 * Estimator implemented for Lshape domain
 */

template <int dim>
class SolutionEstimate : public AmandusIntegrator<dim>
{
public:
    SolutionEstimate(FlowFunction<dim>& solution);
    virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
    virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo,
                          MeshWorker::IntegrationInfo<dim>& info) const;
    virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                      MeshWorker::IntegrationInfo<dim>& info1,
                      MeshWorker::IntegrationInfo<dim>& info2) const;

private:
    SmartPointer<FlowFunction<dim>, SolutionEstimate<dim>> solution;
};

template <int dim>
SolutionEstimate<dim>::SolutionEstimate(FlowFunction<dim>& solution)
    : solution(&solution)

{
    this->use_boundary = true;
    this->use_face = true;
    this->add_flags(update_hessians);
    this->add_flags(update_gradients);
}

template <int dim>
void
SolutionEstimate<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo,
                            MeshWorker::IntegrationInfo<dim>& info) const
{
    std::vector<std::vector<double>> rhs(
                                         dim + 1, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));

    solution->vector_laplacians(info.fe_values(0).get_quadrature_points(), rhs);


    const std::vector<Tensor<2, dim>>&  h0 = info.hessians[0][0] ;
    const std::vector<Tensor<2, dim>>&  h1 = info.hessians[0][1] ;
    const std::vector<Tensor<1, dim>>& p1 = info.gradients[0][dim];

    for (unsigned int k=0;k<info.fe_values(0).n_quadrature_points;++k)
    {
        Tensor<1,dim> r;

        r[0] += dinfo.cell->diameter()*(trace(h0[k])+p1[k][0]-rhs[0][k]);
        r[1] += dinfo.cell->diameter()*(trace(h1[k])+p1[k][1]-rhs[1][k]);

        dinfo.value(0) += (r*r) * info.fe_values(0).JxW(k);
    }

    dinfo.value(0) = std::sqrt(dinfo.value(0));

}

template <int dim>
void
SolutionEstimate<dim>::boundary(MeshWorker::DoFInfo<dim>& dinfo,
                                MeshWorker::IntegrationInfo<dim>& info) const
{
    const FEValuesBase<dim>& fe = info.fe_values();
    std::vector<std::vector<double>> values(
                                         dim + 1, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));
    solution->vector_values(info.fe_values(0).get_quadrature_points(), values);

    const unsigned int deg = fe.get_fe().tensor_degree();
    const double penalty = Laplace::compute_penalty(dinfo, dinfo, deg, deg);

    for (unsigned int k=0; k<fe.n_quadrature_points;++k)
    {
        Tensor<1,dim> diff;
        for (unsigned int d= 0; d < dim ; ++d)
        {
            diff[d] += (info.values[0][d][k]-values[d][k]);

        }
        dinfo.value(0) += penalty * (diff*diff) * fe.JxW(k);
    }
    dinfo.value(0) = std::sqrt(dinfo.value(0));
}

template <int dim>
void
SolutionEstimate<dim>::face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                            MeshWorker::IntegrationInfo<dim>& info1,
                            MeshWorker::IntegrationInfo<dim>& info2) const
{
    const FEValuesBase<dim>& fe = info1.fe_values();
    double h;
    if (dim == 3)
        h = std::sqrt(dinfo1.face->measure());
    else
        h = dinfo1.face->measure();

    const unsigned int deg1 = info1.fe_values().get_fe().tensor_degree();
    const unsigned int deg2 = info2.fe_values().get_fe().tensor_degree();
    const double penalty = 2.*Laplace::compute_penalty(dinfo1, dinfo2, deg1, deg2);
    for (unsigned int k=0; k<fe.n_quadrature_points;++k)
    {
        Tensor<1,dim> diff ;
        Tensor<1,dim> diff1 ;
        for (unsigned int d= 0; d < dim ; ++d)
        {
            diff += (info1.gradients[0][d][k] -info2.gradients[0][d][k])*fe.normal_vector(k)[d];

            diff[d] -= (info1.values[0][dim][k] - info2.values[0][dim][k])*fe.normal_vector(k)[d];

            diff1[d] +=  info1.values[0][d][k]- info2.values[0][d][k];
        }
        dinfo1.value(0) += (penalty* (diff1*diff1)+h * (diff*diff) )* fe.JxW(k);

    }
    dinfo1.value(0) = std::sqrt(dinfo1.value(0));
    dinfo2.value(0) = dinfo1.value(0);
}
}

#endif
