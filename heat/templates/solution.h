/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __laplace_solution_h
#define __laplace_solution_h

#include <amandus/integrator.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

namespace LaplaceIntegrators
{
/**
 * Integrate the right hand side for a Laplace problem, where the
 * solution is given.
 *
 * @ingroup integrators
 */
template <int dim>
class SolutionRHS : public AmandusIntegrator<dim>
{
public:
  SolutionRHS(Function<dim>& solution);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  SmartPointer<Function<dim>, SolutionRHS<dim>> solution;
};

/**
 * Integrate the residual for a Laplace problem, where the
 * solution is given.
 *
 * @ingroup integrators
 */
template <int dim>
class SolutionResidual : public AmandusIntegrator<dim>
{
public:
  SolutionResidual(Function<dim>& solution);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  SmartPointer<Function<dim>, SolutionResidual<dim>> solution;
};

template <int dim>
class SolutionError : public AmandusIntegrator<dim>
{
public:
  SolutionError(Function<dim>& solution);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  SmartPointer<Function<dim>, SolutionError<dim>> solution;
};

template <int dim>
class SolutionEstimate : public AmandusIntegrator<dim>
{
public:
  SolutionEstimate(Function<dim>& solution);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  SmartPointer<Function<dim>, SolutionEstimate<dim>> solution;
};

//----------------------------------------------------------------------//

template <int dim>
SolutionRHS<dim>::SolutionRHS(Function<dim>& solution)
  : solution(&solution)
{
  this->use_boundary = true;
  this->use_face = false;
}

template <int dim>
void
SolutionRHS<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
    rhs[k] = -solution->laplacian(info.fe_values(0).quadrature_point(k));

  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
}

template <int dim>
void
SolutionRHS<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  if (info.fe_values(0).get_fe().conforms(FiniteElementData<dim>::H1))
    return;

  const FEValuesBase<dim>& fe = info.fe_values();
  Vector<double>& local_vector = dinfo.vector(0).block(0);

  std::vector<double> boundary_values(fe.n_quadrature_points);
  solution->value_list(fe.get_quadrature_points(), boundary_values);

  const unsigned int deg = fe.get_fe().tensor_degree();
  const double penalty = 2. * Laplace::compute_penalty(dinfo, dinfo, deg, deg);

  for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      local_vector(i) += (penalty * fe.shape_value(i, k) * boundary_values[k] -
                          (fe.normal_vector(k) * fe.shape_grad(i, k)) * boundary_values[k]) *
                         fe.JxW(k);
}

template <int dim>
void
SolutionRHS<dim>::face(DoFInfo<dim>&, DoFInfo<dim>&, IntegrationInfo<dim>&,
                       IntegrationInfo<dim>&) const
{
}

//----------------------------------------------------------------------//

template <int dim>
SolutionResidual<dim>::SolutionResidual(Function<dim>& solution)
  : solution(&solution)
{
  this->use_boundary = true;
  this->use_face = true;
}

template <int dim>
void
SolutionResidual<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(info.values.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, ExcDimensionMismatch(info.values.size(), 1));

  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
    rhs[k] = -solution->laplacian(info.fe_values(0).quadrature_point(k));

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
SolutionResidual<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  if (info.fe_values(0).get_fe().conforms(FiniteElementData<dim>::H1))
    return;

  std::vector<double> boundary_values(info.fe_values(0).n_quadrature_points, 0.);
  solution->value_list(info.fe_values(0).get_quadrature_points(), boundary_values);

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
SolutionResidual<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                            IntegrationInfo<dim>& info2) const
{
  if (info1.fe_values(0).get_fe().conforms(FiniteElementData<dim>::H1))
    return;

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
SolutionError<dim>::SolutionError(Function<dim>& solution)
  : solution(&solution)
{
  this->use_boundary = false;
  this->use_face = false;
  this->error_types.push_back(2);
  this->error_types.push_back(2);
}

template <int dim>
void
SolutionError<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(dinfo.n_values() >= 2, ExcDimensionMismatch(dinfo.n_values(), 4));

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
  {
    const double dx = info.fe_values(0).JxW(k);

    Tensor<1, dim> Du_h = info.gradients[0][0][k];
    Tensor<1, dim> Du = solution->gradient(info.fe_values(0).quadrature_point(k));
    Tensor<1, dim> ddiff = Du - Du_h;
    double u_h = info.values[0][0][k];
    double u = solution->value(info.fe_values(0).quadrature_point(k));
    double diff = u - u_h;

    // 0. L^2(u)
    dinfo.value(0) += (diff * diff) * dx;
    // 1. H^1(u)
    dinfo.value(1) += (ddiff * ddiff) * dx;
  }
  dinfo.value(0) = std::sqrt(dinfo.value(0));
  dinfo.value(1) = std::sqrt(dinfo.value(1));
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

//----------------------------------------------------------------------//

template <int dim>
SolutionEstimate<dim>::SolutionEstimate(Function<dim>& solution)
  : solution(&solution)
{
  this->use_boundary = true;
  this->use_face = true;
  this->add_flags(update_hessians);
}

template <int dim>
void
SolutionEstimate<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values();

  const std::vector<Tensor<2, dim>>& DDuh = info.hessians[0][0];
  for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
  {
    const double t = dinfo.cell->diameter() *
                     (trace(DDuh[k]) - solution->laplacian(info.fe_values(0).quadrature_point(k)));
    dinfo.value(0) += t * t * fe.JxW(k);
  }
  dinfo.value(0) = std::sqrt(dinfo.value(0));
}

template <int dim>
void
SolutionEstimate<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values();

  std::vector<double> boundary_values(fe.n_quadrature_points, 0.);
  solution->value_list(fe.get_quadrature_points(), boundary_values);

  const std::vector<double>& uh = info.values[0][0];

  const unsigned int deg = fe.get_fe().tensor_degree();
  const double penalty = Laplace::compute_penalty(dinfo, dinfo, deg, deg);

  for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
    dinfo.value(0) +=
      penalty * (boundary_values[k] - uh[k]) * (boundary_values[k] - uh[k]) * fe.JxW(k);
  dinfo.value(0) = std::sqrt(dinfo.value(0));
}

template <int dim>
void
SolutionEstimate<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                            IntegrationInfo<dim>& info2) const
{
  const FEValuesBase<dim>& fe = info1.fe_values();
  const std::vector<double>& uh1 = info1.values[0][0];
  const std::vector<double>& uh2 = info2.values[0][0];
  const std::vector<Tensor<1, dim>>& Duh1 = info1.gradients[0][0];
  const std::vector<Tensor<1, dim>>& Duh2 = info2.gradients[0][0];

  const unsigned int deg1 = info1.fe_values().get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values().get_fe().tensor_degree();
  const double penalty = 2. * Laplace::compute_penalty(dinfo1, dinfo2, deg1, deg2);

  double h;
  if (dim == 3)
    h = std::sqrt(dinfo1.face->measure());
  else
    h = dinfo1.face->measure();

  for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
  {
    double diff1 = uh1[k] - uh2[k];
    double diff2 = fe.normal_vector(k) * Duh1[k] - fe.normal_vector(k) * Duh2[k];
    dinfo1.value(0) += (penalty * diff1 * diff1 + h * diff2 * diff2) * fe.JxW(k);
  }
  dinfo1.value(0) = std::sqrt(dinfo1.value(0));
  dinfo2.value(0) = dinfo1.value(0);
}
}

#endif
