/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __advection_matrix_h
#define __advection_matrix_h

#include <amandus/integrator.h>
#include <deal.II/integrators/advection.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/meshworker/integration_info.h>

using namespace dealii::MeshWorker;

namespace Advection
{
/**
 * Integrator for Advection problems and heat equation.
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
  Matrix(const Parameters& par);

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const override;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const override;

private:
  dealii::SmartPointer<const Parameters, class Matrix<dim>> parameters;
  std::vector<std::vector<double>> direction;
};

template <int dim>
Matrix<dim>::Matrix(const Parameters& par)
  : parameters(&par)
  , direction(dim, std::vector<double>(1))
{
  // this->input_vector_names.push_back("Newton iterate");
  direction[0][0] = 1.;
  direction[1][0] = 2.;
}

template <int dim>
void
Matrix<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  AssertDimension(dinfo.n_matrices(), 1);
  dealii::LocalIntegrators::Advection::cell_matrix(
    dinfo.matrix(0, false).matrix, info.fe_values(0), info.fe_values(0), direction);
}

template <int dim>
void
Matrix<dim>::boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  dealii::LocalIntegrators::Advection::upwind_value_matrix(
    dinfo.matrix(0, false).matrix, info.fe_values(0), info.fe_values(0), direction);
}

template <int dim>
void
Matrix<dim>::face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                  IntegrationInfo<dim>& info2) const
{
  dealii::LocalIntegrators::Advection::upwind_value_matrix(dinfo1.matrix(0, false).matrix,
                                                           dinfo1.matrix(0, true).matrix,
                                                           dinfo2.matrix(0, true).matrix,
                                                           dinfo2.matrix(0, false).matrix,
                                                           info1.fe_values(0),
                                                           info2.fe_values(0),
                                                           info1.fe_values(0),
                                                           info2.fe_values(0),
                                                           direction);
}
}

#endif
