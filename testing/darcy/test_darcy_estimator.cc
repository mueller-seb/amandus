/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#define BOOST_TEST_MODULE test_darcy_estimator.h
#include <boost/test/included/unit_test.hpp>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/quadrature_lib.h>

#include <amandus/darcy/integrators.h>
#include <amandus/darcy/estimator.h>
#include <amandus/darcy/checkerboard/solution.h>

using namespace dealii;
using namespace Darcy;

#define TOL 0.00001

class TestFunction : public Function<2>
{
  public:
    TestFunction();

    virtual double value(const Point<2>& p,
                         const unsigned int component) const;
};

TestFunction::TestFunction() : Function<2>(3)
{}

double TestFunction::value(const Point<2>& /*p*/,
                           const unsigned int component) const
{
  if(component == 0) {
    return 0;
  } else if(component == 1) {
    return 0;
  } 
  return 1;
}


BOOST_AUTO_TEST_CASE(postprocessor_test)
{
  const unsigned int dim = 2;

  Triangulation<dim> tr;
  GridGenerator::hyper_cube(tr, -1, 1);

  IdentityTensorFunction<dim> weight;

  TestFunction test_function;
  FESystem<dim> solution_fe(FE_RaviartThomas<dim>(0), 1,
                            FE_DGQ<dim>(0), 1);
  DoFHandler<dim> solution_dofh;
  solution_dofh.initialize(tr, solution_fe);
  Vector<double> solution(solution_dofh.n_dofs());

  ConstraintMatrix solution_constraints;
  DoFTools::make_hanging_node_constraints(solution_dofh,
                                          solution_constraints);
  solution_constraints.close();

  QGauss<dim> solution_quadrature(solution_fe.tensor_degree() + 2);

  VectorTools::project(solution_dofh,
                       solution_constraints,
                       solution_quadrature,
                       test_function,
                       solution);

  FE_DGQ<dim> pp_fe(0);
  DoFHandler<dim> pp_dofh;
  pp_dofh.initialize(tr, pp_fe);
  Vector<double> pp(pp_dofh.n_dofs());

  Postprocessor<dim> postprocessor(pp_dofh,
                                   solution_dofh,
                                   weight);
  postprocessor.postprocess(pp, solution);

  QGauss<dim> pp_quadrature(pp_fe.tensor_degree() + 2);
  std::vector<Vector<double> > pp_vals(pp_quadrature.size(), Vector<double>(1));
  FEValues<dim> pp_fev(pp_fe, pp_quadrature, 
                       update_values | update_quadrature_points);
  for(typename DoFHandler<dim>::active_cell_iterator cell = pp_dofh.begin_active();
      cell != pp_dofh.end();
      ++cell)
  {
    pp_fev.reinit(cell);
    pp_fev.get_function_values(pp, pp_vals);
    for(unsigned int q = 0; q < pp_quadrature.size(); ++q)
    {
      BOOST_CHECK_CLOSE(pp_vals[q](0), 1, TOL);
    }
  }
                        
  // refine, reproject, repostprocess
  // assert
}


class InterpolationTestFunction : public Function<2>
{
  public:
    virtual double value(const Point<2>& p,
                         const unsigned int component) const;
};

double InterpolationTestFunction::value(const Point<2>& p,
                                        const unsigned int /*component*/) const
{
  if(p(0) > 0.0 && p(1) < 0.0)
  {
    return 0.0;
  }
  return 1.0;
}


BOOST_AUTO_TEST_CASE(interpolator_test)
{
  const unsigned int dim = 2;

  Triangulation<dim> tr;
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global();

  std::vector<double> weight_params;
  weight_params.push_back(1.0/12.0);
  weight_params.push_back(1.0/12.0);
  weight_params.push_back(1.0/12.0);
  weight_params.push_back(3.0/4.0);
  Checkerboard::CheckerboardTensorFunction weight(weight_params);

  InterpolationTestFunction test_function;
  FE_DGQ<dim> input_fe(0);
  DoFHandler<dim> input_dofh;
  input_dofh.initialize(tr, input_fe);
  Vector<double> input(input_dofh.n_dofs());

  ConstraintMatrix input_constraints;
  DoFTools::make_hanging_node_constraints(input_dofh,
                                          input_constraints);
  input_constraints.close();

  QGauss<dim> input_quadrature(input_fe.tensor_degree() + 2);

  VectorTools::project(input_dofh,
                       input_constraints,
                       input_quadrature,
                       test_function,
                       input);


  Interpolator<dim> interpolator(input_dofh,
                                 input_fe.tensor_degree() + 2,
                                 weight);
  Vector<double> interpolation;
  interpolator.init_vector(interpolation);
  interpolator.interpolate(interpolation, input);

  Functions::FEFieldFunction<dim> interpolation_function(interpolator.get_dofh(),
                                                         interpolation);
  BOOST_CHECK_CLOSE(interpolation_function.value(Point<dim>(0.0, 0.0)), 0.5, TOL);
}
