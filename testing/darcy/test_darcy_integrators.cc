/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#define BOOST_TEST_MODULE test_darcy_integrators.h
#include <boost/test/included/unit_test.hpp>

#include <amandus/darcy/integrators.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

using namespace dealii;
using namespace MeshWorker;

#define TOL 0.00001

template <int dim>
class IdentityTensorFunction : public TensorFunction<2, dim>
{
public:
  typedef typename TensorFunction<2, dim>::value_type value_type;
  IdentityTensorFunction();

  virtual value_type value(const Point<dim>& p) const;

  Tensor<2, dim> identity;
};

template <int dim>
IdentityTensorFunction<dim>::IdentityTensorFunction()
{
  for (unsigned int i = 0; i < dim; ++i)
  {
    identity[i][i] = 1.0;
  }
}

template <int dim>
typename IdentityTensorFunction<dim>::value_type
IdentityTensorFunction<dim>::value(const Point<dim>& /*p*/) const
{
  return identity;
}

template <int dim>
class CoefficientIntegrator : public LocalIntegrator<dim>
{
public:
  CoefficientIntegrator();

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const;

private:
  IdentityTensorFunction<dim> identity;
};

template <int dim>
CoefficientIntegrator<dim>::CoefficientIntegrator()
{
  this->use_boundary = false;
  this->use_face = false;
}

template <int dim>
void
CoefficientIntegrator<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  Darcy::weighted_mass_matrix(dinfo.matrix(0).matrix, info.fe_values(0), identity);
}

template <int dim>
void
CoefficientIntegrator<dim>::boundary(DoFInfo<dim>& /*dinfo*/, IntegrationInfo<dim>& /*info*/) const
{
}

template <int dim>
void
CoefficientIntegrator<dim>::face(DoFInfo<dim>& /*dinfo1*/, DoFInfo<dim>& /*dinfo2*/,
                                 IntegrationInfo<dim>& /*info1*/,
                                 IntegrationInfo<dim>& /*info2*/) const
{
}

template <int dim>
class ReferenceIntegrator : public LocalIntegrator<dim>
{
public:
  ReferenceIntegrator();

  virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
  virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2, IntegrationInfo<dim>& info1,
                    IntegrationInfo<dim>& info2) const;

private:
  IdentityTensorFunction<dim> identity;
};

template <int dim>
ReferenceIntegrator<dim>::ReferenceIntegrator()
{
  this->use_boundary = false;
  this->use_face = false;
}

template <int dim>
void
ReferenceIntegrator<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
{
  LocalIntegrators::L2::mass_matrix(dinfo.matrix(0).matrix, info.fe_values(0));
}

template <int dim>
void
ReferenceIntegrator<dim>::boundary(DoFInfo<dim>& /*dinfo*/, IntegrationInfo<dim>& /*info*/) const
{
}

template <int dim>
void
ReferenceIntegrator<dim>::face(DoFInfo<dim>& /*dinfo1*/, DoFInfo<dim>& /*dinfo2*/,
                               IntegrationInfo<dim>& /*info1*/,
                               IntegrationInfo<dim>& /*info2*/) const
{
}

BOOST_AUTO_TEST_CASE(coefficient_parameters)
{
  const unsigned int dim = 2;

  // finite element
  FESystem<dim> fe(FE_Q<dim>(1), dim);

  // domain and triangulation
  Triangulation<2> tr;
  GridGenerator::hyper_cube(tr, -1, 1);

  // coupling between finite element and triangulation
  DoFHandler<dim> dof_handler(tr);
  dof_handler.distribute_dofs(fe);

  // mapping of reference cell to physical cell
  MappingQ1<dim> mapping;

  // infrastructure for integration
  IntegrationInfoBox<dim> info_box;

  // set update flags
  info_box.initialize_update_flags();
  UpdateFlags update_flags = update_values | update_quadrature_points;
  info_box.add_update_flags_cell(update_flags);

  // initialize fevalues
  info_box.initialize(fe, mapping);

  // initialize dofinfo
  DoFInfo<dim> dof_info(dof_handler);

  // matrix to assemble
  FullMatrix<double> matrix(dof_handler.n_dofs());

  Assembler::MatrixSimple<FullMatrix<double>> assembler;
  assembler.initialize(matrix);

  // integrators
  CoefficientIntegrator<dim> coefficient_integrator;
  ReferenceIntegrator<dim> integrator;

  // start assembly
  integration_loop<dim, dim>(dof_handler.begin_active(),
                             dof_handler.end(),
                             dof_info,
                             info_box,
                             coefficient_integrator,
                             assembler);
  // save result
  FullMatrix<double> coefficient_result(matrix);
  matrix = 0;

  // start assembly
  integration_loop<dim, dim>(
    dof_handler.begin_active(), dof_handler.end(), dof_info, info_box, integrator, assembler);

  // compare results
  BOOST_CHECK_EQUAL(coefficient_result.m(), matrix.m());
  BOOST_CHECK_EQUAL(coefficient_result.n(), matrix.n());
  for (unsigned int m = 0; m < coefficient_result.m(); ++m)
  {
    for (unsigned int n = 0; n < coefficient_result.n(); ++n)
    {
      BOOST_CHECK_CLOSE(coefficient_result[m][n], matrix[m][n], TOL);
    }
  }
}
