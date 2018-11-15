/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __heat_rhs_heat_h
#define __heat_rhs_heat_h

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

template <int dim>
class Force : public Function<dim>
{
public:
  Force();
  virtual double value(const Point<dim>& p, const unsigned int component) const override;
  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component) const override;
};

template <int dim>
Force<dim>::Force()
{
}

template <int dim>
double
Force<dim>::value(const Point<dim>& p, const unsigned int component) const
{
  double x = p(0);
  double y = p(1);
  double result = 0;
  if ((component == 0) || (abs(y) < 1e-6))
  {
  if (x < 0)
	result = 1;
  if (x > 0)
	result = -1;
  }

  return result;
}

template <int dim>
void
Force<dim>::value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                         const unsigned int) const
{
  AssertDimension(points.size(), values.size());

  for (unsigned int k = 0; k < points.size(); ++k)
  {
    const Point<dim>& p = points[k];
    values[k] = 1. * p(0);
  }
}


template <int dim>
class RHS : public AmandusIntegrator<dim>
{
public:
  RHS();

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;
};

template <int dim>
class Estimate : public AmandusIntegrator<dim>
{
public:
  Estimate();

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;
};

//----------------------------------------------------------------------//





//-----//

template <int dim>
RHS<dim>::RHS()
{
  this->use_boundary = false;
  this->use_face = true;
}

template <int dim>
void
RHS<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);
  Force<dim> f;

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
    rhs[k] = f.value(info.fe_values(0).quadrature_point(k), 0);

  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
}

template <int dim>
void
RHS<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
/*  if (info.fe_values(0).get_fe().conforms(FiniteElementData<dim>::H1))
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
                         fe.JxW(k);*/

}

template <int dim>
void
RHS<dim>::face(DoFInfo<dim>& dinfo, DoFInfo<dim>&, IntegrationInfo<dim>& info,
                       IntegrationInfo<dim>&) const
{
  std::vector<double> rhs(info.fe_values(0).n_quadrature_points, 0.);
  Force<dim> f;

  for (unsigned int k = 0; k < info.fe_values(0).n_quadrature_points; ++k)
    rhs[k] = f.value(info.fe_values(0).quadrature_point(k), 1);

  L2::L2(dinfo.vector(0).block(0), info.fe_values(0), rhs);
}

//----------------------------------------------------------------------//


template <int dim>
Estimate<dim>::Estimate()
{
  this->use_boundary = true;
  this->use_face = true;
  this->add_flags(update_hessians);
}

template <int dim>
void
Estimate<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values();
  Force<dim> f;

  const std::vector<Tensor<2, dim>>& DDuh = info.hessians[0][0];
  for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
  {
    const double t = dinfo.cell->diameter() *
                     (trace(DDuh[k]) + f.value(info.fe_values(0).quadrature_point(k), 0));
    dinfo.value(0) += t * t * fe.JxW(k);
  }
  dinfo.value(0) = std::sqrt(dinfo.value(0));
}

template <int dim>
void
Estimate<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values();

  std::vector<double> boundary_values(fe.n_quadrature_points, 0.); //vector of zero, boundary values = 0
  //solution->value_list(fe.get_quadrature_points(), boundary_values);

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
void
Estimate<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
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
    h = dinfo1.face->measure();
 */

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
