/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __readiff_matrix_h
#define __readiff_matrix_h

#include <amandus/integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/integration_info.h>
#include <readiff/parameters.h>

using namespace dealii;
using namespace LocalIntegrators;

/**
 * Integrators for ReactionDiffusion problems
 *
 * These integrators deal with the equations
 *
 * \f{align*}{
 * 0 &= u' - \alpha_1\Delta u + A_1+B_1u+C_1v+D_1uv+E_1u^2+F_1v^2+G_1u^2v+H_1uv^2 \\
 * 0 &= v' - \alpha_2\Delta v + A_2+B_2u+C_2v+D_2uv+E_2u^2+F_2v^2+G_2u^2v+H_2uv^2
 * \f}
 *
 */
namespace ReactionDiffusion
{
/**
 * The derivative of the residual operator in ImplicitResidual
 * consists of 4 matrices, namely:
 *
 */
template <int dim>
class Matrix : public AmandusIntegrator<dim>
{
public:
  Matrix(const Parameters& par);

  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const override;
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo,
                        MeshWorker::IntegrationInfo<dim>& info) const override;
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                    MeshWorker::IntegrationInfo<dim>& info1,
                    MeshWorker::IntegrationInfo<dim>& info2) const override;

private:
  SmartPointer<const Parameters, class Matrix<dim>> parameters;
};

template <int dim>
Matrix<dim>::Matrix(const Parameters& par)
  : parameters(&par)
{
  this->use_boundary = false;
}

template <int dim>
void
Matrix<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  AssertDimension(dinfo.n_matrices(), 4);
  //    Assert (info.values.size() >0, ExcInternalError());

  const Parameters& p = *parameters;
  Laplace::cell_matrix(dinfo.matrix(0, false).matrix, info.fe_values(0), p.alpha1);
  Laplace::cell_matrix(dinfo.matrix(3, false).matrix, info.fe_values(0), p.alpha2);
  if (info.values.size() > 0)
  {
    AssertDimension(info.values[0].size(), 2);
    AssertDimension(info.values[0][0].size(), info.fe_values(0).n_quadrature_points);
    AssertDimension(info.values[0][1].size(), info.fe_values(0).n_quadrature_points);
    std::vector<double> Du_ru(info.fe_values(0).n_quadrature_points);
    std::vector<double> Dv_ru(info.fe_values(0).n_quadrature_points);
    std::vector<double> Dv_rv(info.fe_values(0).n_quadrature_points);
    std::vector<double> Du_rv(info.fe_values(0).n_quadrature_points);
    for (unsigned int k = 0; k < Du_ru.size(); ++k)
    {
      const double u = info.values[0][0][k];
      const double v = info.values[0][1][k];
      Du_ru[k] = p.B1 + p.D1 * v + 2. * p.E1 * u + 2. * p.G1 * u * v + p.H1 * v * v;
      Dv_ru[k] = p.C1 + p.D1 * u + 2. * p.F1 * v + 2. * p.H1 * u * v + p.G1 * u * u;
      Dv_rv[k] = p.C2 + p.D2 * u + 2. * p.F2 * v + 2. * p.H2 * u * v + p.G2 * u * u;
      Du_rv[k] = p.B2 + p.D2 * v + 2. * p.E2 * u + 2. * p.G2 * u * v + p.H2 * v * v;
    }
    L2::weighted_mass_matrix(dinfo.matrix(0, false).matrix, info.fe_values(0), Du_ru);
    L2::weighted_mass_matrix(dinfo.matrix(1, false).matrix, info.fe_values(0), Dv_ru);
    L2::weighted_mass_matrix(dinfo.matrix(2, false).matrix, info.fe_values(0), Du_rv);
    L2::weighted_mass_matrix(dinfo.matrix(3, false).matrix, info.fe_values(0), Dv_rv);
  }
}

template <int dim>
void
Matrix<dim>::boundary(MeshWorker::DoFInfo<dim>&, typename MeshWorker::IntegrationInfo<dim>&) const
{
}

template <int dim>
void
Matrix<dim>::face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                  MeshWorker::IntegrationInfo<dim>& info1,
                  MeshWorker::IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  const Parameters& p = *parameters;
  Laplace::ip_matrix(dinfo1.matrix(0, false).matrix,
                     dinfo1.matrix(0, true).matrix,
                     dinfo2.matrix(0, true).matrix,
                     dinfo2.matrix(0, false).matrix,
                     info1.fe_values(0),
                     info2.fe_values(0),
                     Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
                     p.alpha1);
  Laplace::ip_matrix(dinfo1.matrix(3, false).matrix,
                     dinfo1.matrix(3, true).matrix,
                     dinfo2.matrix(3, true).matrix,
                     dinfo2.matrix(3, false).matrix,
                     info1.fe_values(0),
                     info2.fe_values(0),
                     Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
                     p.alpha2);
}
}

#endif
