/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __heat_rhs_h
#define __heat_rhs_h

#include <amandus/integrator.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/tensor.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

namespace HeatIntegrators
{
/**
 * Integrate the right hand side of the Heat problem, where the
 * solution is given.
 *
 * @ingroup integrators
 */
template <int dim>
class RHS : public AmandusIntegrator<dim>
{
public:
  RHS(Function<dim>& f);
  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  SmartPointer<Function<dim>, RHS<dim>> f;
};

/**
 * Integrate the error for the Heat problem, where the
 * solution is given.
 *
 * @ingroup integrators
 */
template <int dim>
class Error : public AmandusIntegrator<dim>
{
public:
  Error(Function<dim>& solution);
  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  SmartPointer<Function<dim>, Error<dim>> solution;
};

/*
 * Estimator of the Laplace example
 *
 * @ingroup integrators
*/
template <int dim>
class Estimate : public AmandusIntegrator<dim>
{
public:
  Estimate(Function<dim>& f);
  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  SmartPointer<Function<dim>, Estimate<dim>> f;
};
/*
 * This ist the (Kelly?)-Estimator presented in deal.ii Tutorial 39
 *
 * @ingroup integrators
*/
template <int dim>
class EstimateT : public AmandusIntegrator<dim>
{
public:
  EstimateT(Function<dim>& f);
  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  SmartPointer<Function<dim>, EstimateT<dim>> f;
};

//----------------------------------------------------------------------//

template <int dim>
RHS<dim>::RHS(Function<dim>& f)
     : f(&f)
{
  this->use_boundary = false;
  this->use_face = true;
}

template <int dim>
void
RHS<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
    rhs[k] = f->value(info.fe_values(0).quadrature_point(k), 0);

  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
}

template <int dim>
void
RHS<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
}

template <int dim>
void
RHS<dim>::face(DoFInfo<dim>& dinfo, DoFInfo<dim>&, IntegrationInfo<dim>& info,
                       IntegrationInfo<dim>&) const
{
const FEValuesBase<dim>& fe = info.fe_values(0);
double ydiff = fe.quadrature_point(fe.n_quadrature_points-1)(1)-fe.quadrature_point(0)(1);

if (abs(ydiff) < 1e-5) //involve horizontal faces only
  {
  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
    rhs[k] = f->value(info.fe_values(0).quadrature_point(k), 1);

  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
  }
}

//----------------------------------------------------------------------//


template <int dim>
Error<dim>::Error(Function<dim>& solution)
  : solution(&solution)
{
  this->use_boundary = false;
  this->use_face = false;
  this->error_types.push_back(2);
  this->error_types.push_back(2);
}

template <int dim>
void Error<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Assert(dinfo.n_values() >= 2, ExcDimensionMismatch(dinfo.n_values(), 4));

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
  {
    const double dx = info.fe_values(0).JxW(k);

    Tensor<1, dim> Du_h = info.gradients[0][0][k];
    Tensor<1, dim> Du = solution->gradient(info.fe_values(0).quadrature_point(k));
    Tensor<1, dim> ddiff = Du - Du_h;
    double u_h = info.values[0][0][k];
    double u = solution->value(info.fe_values(0).quadrature_point(k), 0);
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
void Error<dim>::boundary(DoFInfo<dim>&, IntegrationInfo<dim>&) const
{
}

template <int dim>
void Error<dim>::face(DoFInfo<dim>&, DoFInfo<dim>&, IntegrationInfo<dim>&,
                         IntegrationInfo<dim>&) const
{
}

//----------------------------------------------------------------------//

template <int dim>
Estimate<dim>::Estimate(Function<dim>& f)
  : f(&f)
{
  this->use_boundary = true;
  this->use_face = true;
  this->add_flags(update_hessians);
}

template <int dim>
void Estimate<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values();

  const std::vector<Tensor<2, dim>>& DDuh = info.hessians[0][0];
  for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
  {
    const double t = dinfo.cell->diameter() *
                     (trace(DDuh[k]) - f->value(info.fe_values(0).quadrature_point(k), 0));
    dinfo.value(0) += t * t * fe.JxW(k);
  }
  dinfo.value(0) = std::sqrt(dinfo.value(0));
}

template <int dim>
void Estimate<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values();

  std::vector<double> boundary_values(fe.n_quadrature_points, 0.);
  //solution->value_list(fe.get_quadrature_points(), boundary_values);

  const std::vector<double>& uh = info.values[0][0];

  const unsigned int deg = fe.get_fe().tensor_degree();
  const double penalty = Laplace::compute_penalty(dinfo, dinfo, deg, deg);

  for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
    dinfo.value(0) +=
      penalty * (boundary_values[k] - uh[k]) * (boundary_values[k] - uh[k]) * fe.JxW(k);
  dinfo.value(0) = std::sqrt(dinfo.value(0));
}

template <int dim>
void Estimate<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
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


//----------------------------------------------------------------------//

template <int dim>
EstimateT<dim>::EstimateT(Function<dim>& f)
   : f(&f)
{
  this->use_boundary = true;
  this->use_face = true;
  this->add_flags(update_hessians);
}

template <int dim>
void EstimateT<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values();

  const std::vector<Tensor<2, dim>>& DDuh = info.hessians[0][0];
  for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
  {
    const double t = dinfo.cell->diameter() *
                     (trace(DDuh[k]) + f->value(info.fe_values(0).quadrature_point(k), 0));
    dinfo.value(0) += t * t * fe.JxW(k);
  }
  dinfo.value(0) = std::sqrt(dinfo.value(0));
}

template <int dim>
void EstimateT<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values();

  std::vector<double> boundary_values(fe.n_quadrature_points, 0.); //vector of zeros
  //f->value_list(fe.get_quadrature_points(), boundary_values);

  const std::vector<double>& uh = info.values[0][0];

  const unsigned int deg = fe.get_fe().tensor_degree();

  //const double penalty = Laplace::compute_penalty(dinfo, dinfo, deg, deg);

  const double penalty = 2. * deg * (deg+1) * dinfo.face->measure() / dinfo.cell->measure(); //from Tutorial 39

  for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
    dinfo.value(0) +=
      penalty * (boundary_values[k] - uh[k]) * (boundary_values[k] - uh[k]) * fe.JxW(k);
  dinfo.value(0) = std::sqrt(dinfo.value(0));
}

template <int dim> //adapted to tutorial 39
void EstimateT<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                            IntegrationInfo<dim>& info2) const
{
  const FEValuesBase<dim>& fe = info1.fe_values();
  const std::vector<double>& uh1 = info1.values[0][0];
  const std::vector<double>& uh2 = info2.values[0][0];
  const std::vector<Tensor<1, dim>>& Duh1 = info1.gradients[0][0];
  const std::vector<Tensor<1, dim>>& Duh2 = info2.gradients[0][0];

/*
  const unsigned int deg1 = info1.fe_values().get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values().get_fe().tensor_degree();
  const double penalty = 2. * Laplace::compute_penalty(dinfo1, dinfo2, deg1, deg2);

  double h;
  if (dim == 3)
    h = std::sqrt(dinfo1.face->measure());
  else
    h = dinfo1.face->measure();*/

  const unsigned int deg = fe.get_fe().tensor_degree();
  const double penalty1 = deg * (deg+1) * dinfo1.face->measure() / dinfo1.cell->measure();
  const double penalty2 = deg * (deg+1) * dinfo2.face->measure() / dinfo2.cell->measure();
  const double penalty = penalty1 + penalty2;
  const double h = dinfo1.face->measure();
  for (unsigned k=0; k<fe.n_quadrature_points; ++k)
    {
      double diff1 = uh1[k] - uh2[k];
      double diff2 = fe.normal_vector(k) * Duh1[k] - fe.normal_vector(k) * Duh2[k];
      dinfo1.value(0) += (penalty * diff1*diff1 + h * diff2*diff2) * fe.JxW(k);
    }
  dinfo1.value(0) = std::sqrt(dinfo1.value(0));
  dinfo2.value(0) = dinfo1.value(0);
}
}

#endif
