/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __advectiondiffusion_matrix_h
#define __advectiondiffusion_matrix_h

#include <amandus/integrator.h>
#include <deal.II/integrators/advection.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/meshworker/integration_info.h>

namespace AdvectionDiffusion
{
using namespace dealii;
using namespace MeshWorker;
using namespace LocalIntegrators;

/**
 * Integrator for Advection-Diffusion problems.
 *
 * The distinction between stationary and instationary problems is
 * made by the variable AmandusIntegrator::timestep, which is
 * inherited from the base class. If this variable is zero, we solve a
 * stationary problem. If it is nonzero, we assemble for an implicit
 * scheme.
 *
 * @ingroup integrators
 */
template <int dim>
class Matrix : public AmandusIntegrator<dim>
{
public:
  Matrix(const Parameters& par, double factor1, double factor2,
         std::vector<std::vector<double>> direction, double x1, double x2, double y1, double y2);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  dealii::SmartPointer<const Parameters, class Matrix<dim>> parameters;
  double factor1;
  double factor2;
  std::vector<std::vector<double>> direction;
  double x1;
  double x2;
  double y1;
  double y2;
};

template <int dim>
Matrix<dim>::Matrix(const Parameters& par, double factor1, double factor2,
                    std::vector<std::vector<double>> direction, double x1, double x2, double y1,
                    double y2)
  : parameters(&par)
  , factor1(factor1)
  , factor2(factor2)
  , direction(direction)
  , x1(x1)
  , x2(x2)
  , y1(y1)
  , y2(y2)
{
  // this->input_vector_names.push_back("Newton iterate");
}

template <int dim>
void
Matrix<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  FullMatrix<double>& M1 = dinfo.matrix(0, false).matrix;
  FullMatrix<double>& M2 = dinfo.matrix(0, false).matrix;

  AssertDimension(dinfo.n_matrices(), 1);
  dealii::LocalIntegrators::Advection::cell_matrix(
    M1, info.fe_values(0), info.fe_values(0), direction);

  if (x1 < dinfo.cell->center()[0] && dinfo.cell->center()[0] < x2 &&
      y1 < dinfo.cell->center()[1] && dinfo.cell->center()[1] < y2)
    Laplace::cell_matrix(M2, info.fe_values(0), factor2);
  else
    Laplace::cell_matrix(M2, info.fe_values(0), factor1);
}

template <int dim>
void
Matrix<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{

  FullMatrix<double>& M1 = dinfo.matrix(0, false).matrix;
  FullMatrix<double>& M2 = dinfo.matrix(0, false).matrix;

  dealii::LocalIntegrators::Advection::upwind_value_matrix(
    M1, info.fe_values(0), info.fe_values(0), direction);

  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();

  Point<dim> dir;
  dir(0) = direction[0][0];
  dir(1) = direction[1][0];

  const Tensor<1, dim>& normal = info.fe_values(0).normal_vector(1);

  // Dirichlet boundary condition only at the inflow boundary
  if (normal * dir < 0)
  {
    Laplace::nitsche_matrix(
      M2, info.fe_values(0), Laplace::compute_penalty(dinfo, dinfo, deg, deg), factor1);
  }
}

template <int dim>
void
Matrix<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                  IntegrationInfo<dim>& info2) const
{

  FullMatrix<double>& M11 = dinfo1.matrix(0, false).matrix;
  FullMatrix<double>& M12 = dinfo1.matrix(0, true).matrix;
  FullMatrix<double>& M13 = dinfo2.matrix(0, true).matrix;
  FullMatrix<double>& M14 = dinfo2.matrix(0, false).matrix;
  FullMatrix<double>& M21 = dinfo1.matrix(0, false).matrix;
  FullMatrix<double>& M22 = dinfo1.matrix(0, true).matrix;
  FullMatrix<double>& M23 = dinfo2.matrix(0, true).matrix;
  FullMatrix<double>& M24 = dinfo2.matrix(0, false).matrix;

  dealii::LocalIntegrators::Advection::upwind_value_matrix(M11,
                                                           M12,
                                                           M13,
                                                           M14,
                                                           info1.fe_values(0),
                                                           info2.fe_values(0),
                                                           info1.fe_values(0),
                                                           info2.fe_values(0),
                                                           direction);

  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();

  if (x1 < dinfo1.cell->center()[0] && dinfo1.cell->center()[0] < x2 &&
      y1 < dinfo1.cell->center()[1] && dinfo1.cell->center()[1] < y2 &&
      x1 < dinfo2.cell->center()[0] && dinfo2.cell->center()[0] < x2 &&
      y1 < dinfo2.cell->center()[1] && dinfo2.cell->center()[1] < y2)
    Laplace::ip_matrix(M21,
                       M22,
                       M23,
                       M24,
                       info1.fe_values(0),
                       info2.fe_values(0),
                       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
                       factor2);
  else
    Laplace::ip_matrix(M21,
                       M22,
                       M23,
                       M24,
                       info1.fe_values(0),
                       info2.fe_values(0),
                       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
                       factor1);
}
}

#endif
