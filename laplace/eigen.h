/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __matrix_laplace_h
#define __matrix_laplace_h

#include <amandus/integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/integration_info.h>

using namespace dealii;
using namespace LocalIntegrators;

/**
 * Integrators for Laplace problems.
 *
 * @ingroup integrators
 */
namespace LaplaceIntegrators
{
template <int dim>
class Eigen : public AmandusIntegrator<dim>
{
public:
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo,
                        MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                    MeshWorker::IntegrationInfo<dim>& info1,
                    MeshWorker::IntegrationInfo<dim>& info2) const;
};

template <int dim>
void
Eigen<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  Laplace::cell_matrix(dinfo.matrix(0, false).matrix, info.fe_values(0));
  if (dinfo.n_matrices() == 2)
    L2::mass_matrix(dinfo.matrix(1, false).matrix, info.fe_values(0));
}

template <int dim>
void
Eigen<dim>::boundary(MeshWorker::DoFInfo<dim>& dinfo,
                     typename MeshWorker::IntegrationInfo<dim>& info) const
{
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_matrix(dinfo.matrix(0, false).matrix,
                          info.fe_values(0),
                          Laplace::compute_penalty(dinfo, dinfo, deg, deg));
}

template <int dim>
void
Eigen<dim>::face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                 MeshWorker::IntegrationInfo<dim>& info1,
                 MeshWorker::IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Laplace::ip_matrix(dinfo1.matrix(0, false).matrix,
                     dinfo1.matrix(0, true).matrix,
                     dinfo2.matrix(0, true).matrix,
                     dinfo2.matrix(0, false).matrix,
                     info1.fe_values(0),
                     info2.fe_values(0),
                     Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
}

/*
 * Estimator implemented for real eigenvalues only
 */
template <int dim>
class EigenEstimate : public AmandusIntegrator<dim>
{
public:
  EigenEstimate();
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo,
                        MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                    MeshWorker::IntegrationInfo<dim>& info1,
                    MeshWorker::IntegrationInfo<dim>& info2) const;

private:
  virtual void extract_data(const dealii::AnyData& data);
    double ev;
};

template <int dim>
EigenEstimate<dim>::EigenEstimate()
  : ev(0.)
{
  this->use_boundary = true;
  this->use_face = true;
  this->add_flags(update_hessians);
}

template <int dim>
inline void
EigenEstimate<dim>::extract_data(const dealii::AnyData& data)
{
  const double* tmp = data.try_read_ptr<double>("ev");
  if (tmp != 0)
  {
    ev = *tmp;
  }
}

template <int dim>
void
EigenEstimate<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo,
                         MeshWorker::IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values();

  const std::vector<double>& uh = info.values[0][0];
  const std::vector<Tensor<2, dim>>& DDuh = info.hessians[0][0];

  for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
  {
    const double t = dinfo.cell->diameter() * (trace(DDuh[k]) + ev * uh[k]);
    dinfo.value(0) += t * t * fe.JxW(k);
  }
  dinfo.value(0) = std::sqrt(dinfo.value(0));
}

template <int dim>
void
EigenEstimate<dim>::boundary(MeshWorker::DoFInfo<dim>& dinfo,
                             MeshWorker::IntegrationInfo<dim>& info) const
{
  const FEValuesBase<dim>& fe = info.fe_values();
  const std::vector<double>& uh = info.values[0][0];

  const unsigned int deg = fe.get_fe().tensor_degree();
  const double penalty = Laplace::compute_penalty(dinfo, dinfo, deg, deg);

  for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
    dinfo.value(0) += penalty * uh[k] * uh[k] * fe.JxW(k);
  dinfo.value(0) = std::sqrt(dinfo.value(0));
}

template <int dim>
void
EigenEstimate<dim>::face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                         MeshWorker::IntegrationInfo<dim>& info1,
                         MeshWorker::IntegrationInfo<dim>& info2) const
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