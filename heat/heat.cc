/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * <ul>
 * <li> Stationary Poisson equations</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Exact solution</li>
 * <li> Adaptive linear solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @author Joscha Gedicke
 *
 * @ingroup Laplacegroup
 */

#include <amandus/adaptivity.h>
#include <amandus/apps.h>
#include <amandus/heat/matrix.h>
#include <amandus/heat/solution.h>
#include <amandus/refine_strategy.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <amandus/heat/rhs_one.h>
#include <amandus/heat/RHS_heat.h>

#include <boost/scoped_ptr.hpp>


class Waterfall : public Function<2>
{
public:
  Waterfall(const double k = 60.);

  virtual double value(const Point<2>& p, const unsigned int component = 0) const;

  virtual void value_list(const std::vector<Point<2>>& points, std::vector<double>& values,
                          const unsigned int component = 0) const;
  virtual Tensor<1, 2> gradient(const Point<2>& p, const unsigned int component = 0) const;

  virtual double laplacian(const Point<2>& p, const unsigned int component = 0) const;

private:
  const double k;
};

Waterfall::Waterfall(const double k)
  : k(k)
{
}

double
Waterfall::value(const Point<2>& p, const unsigned int) const
{
  const double x = p[0];
  const double y = p[1];

  return x * y * atan(k * (sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) *
                             (1.0 / 4.0) -
                           1.0)) *
         (x - 1.0) * (y - 1.0);
}

void
Waterfall::value_list(const std::vector<Point<2>>& points, std::vector<double>& values,
                      const unsigned int) const
{
  Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));

  for (unsigned int i = 0; i < points.size(); ++i)
  {
    const Point<2>& p = points[i];
    values[i] = value(p);
  }
}

double
Waterfall::laplacian(const Point<2>& p, const unsigned int) const
{
  const double x = p[0];
  const double y = p[1];

  return -(
    x * atan(k *
             (sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
              1.0)) *
      (x - 1.0) * -2.0 -
    y * atan(k *
             (sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
              1.0)) *
      (y - 1.0) * 2.0 -
    (k * x * y * (x - 1.0) * (y - 1.0) * 1.0 /
     sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * 8.0) /
      ((k * k) *
         pow(sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
               1.0,
             2.0) +
       1.0) -
    (k * x * y * (y * 3.2E1 + 8.0) * (x - 1.0) * 1.0 /
     sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0)) /
      ((k * k) *
         pow(sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
               1.0,
             2.0) +
       1.0) -
    (k * x * y * (x * 3.2E1 - 4.0E1) * (y - 1.0) * 1.0 /
     sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0)) /
      ((k * k) *
         pow(sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
               1.0,
             2.0) +
       1.0) -
    (k * x * (y * 3.2E1 + 8.0) * (x - 1.0) * (y - 1.0) * 1.0 /
     sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0)) /
      ((k * k) *
         pow(sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
               1.0,
             2.0) +
       1.0) -
    (k * y * (x * 3.2E1 - 4.0E1) * (x - 1.0) * (y - 1.0) * 1.0 /
     sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0)) /
      ((k * k) *
         pow(sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
               1.0,
             2.0) +
       1.0) +
    (k * x * y * pow(x * 3.2E1 - 4.0E1, 2.0) * (x - 1.0) * (y - 1.0) * 1.0 /
     pow(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1, 3.0 / 2.0) *
     (1.0 / 1.6E1)) /
      ((k * k) *
         pow(sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
               1.0,
             2.0) +
       1.0) +
    (k * x * y * pow(y * 3.2E1 + 8.0, 2.0) * (x - 1.0) * (y - 1.0) * 1.0 /
     pow(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1, 3.0 / 2.0) *
     (1.0 / 1.6E1)) /
      ((k * k) *
         pow(sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
               1.0,
             2.0) +
       1.0) +
    ((k * k * k) * x * y * pow(x * 3.2E1 - 4.0E1, 2.0) * 1.0 /
     pow((k * k) * pow(sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) *
                           (1.0 / 4.0) -
                         1.0,
                       2.0) +
           1.0,
         2.0) *
     (sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) - 1.0) *
     (x - 1.0) * (y - 1.0) * (1.0 / 3.2E1)) /
      (x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) +
    ((k * k * k) * x * y * pow(y * 3.2E1 + 8.0, 2.0) * 1.0 /
     pow((k * k) * pow(sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) *
                           (1.0 / 4.0) -
                         1.0,
                       2.0) +
           1.0,
         2.0) *
     (sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) - 1.0) *
     (x - 1.0) * (y - 1.0) * (1.0 / 3.2E1)) /
      (x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1));
}

Tensor<1, 2>
Waterfall::gradient(const Point<2>& p, const unsigned int) const
{
  Tensor<1, 2> val;
  const double x = p[0];
  const double y = p[1];

  val[0] =
    y * atan(k *
             (sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
              1.0)) *
      (x - 1.0) * (y - 1.0) +
    x * y * atan(k * (sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) *
                        (1.0 / 4.0) -
                      1.0)) *
      (y - 1.0) +
    (k * x * y * (x * 3.2E1 - 4.0E1) * (x - 1.0) * (y - 1.0) * 1.0 /
     sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 8.0)) /
      ((k * k) *
         pow(sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
               1.0,
             2.0) +
       1.0);
  val[1] =
    x * atan(k *
             (sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
              1.0)) *
      (x - 1.0) * (y - 1.0) +
    x * y * atan(k * (sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) *
                        (1.0 / 4.0) -
                      1.0)) *
      (x - 1.0) +
    (k * x * y * (y * 3.2E1 + 8.0) * (x - 1.0) * (y - 1.0) * 1.0 /
     sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 8.0)) /
      ((k * k) *
         pow(sqrt(x * -4.0E1 + y * 8.0 + (x * x) * 1.6E1 + (y * y) * 1.6E1 + 2.6E1) * (1.0 / 4.0) -
               1.0,
             2.0) +
       1.0);

  return val;
}


//---------------------------------------------------------//

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  AmandusParameters param;
  param.declare_entry("MaxDofs", "1000", Patterns::Integer());
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, 0, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  Waterfall exact_solution;

  HeatIntegrators::Matrix<d> matrix_integrator;
//HeatIntegrators::SolutionRHS<d> rhs_integrator(exact_solution);
//HeatIntegrators::SolutionError<d> error_integrator(exact_solution);
 HeatIntegrators::RHS<d> rhs_integrator(exact_solution);
//HeatIntegrators::SolutionEstimate<d> estimate_integrator(exact_solution);

  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  RefineStrategy::MarkBulk<d> refine_strategy(tr, 0.5);

  /*adaptive_refinement_linear_loop(param.get_integer("MaxDofs"),
                                  app,
                                  tr,
                                  solver,
                                  residual,
                                  estimate_integrator,
                                  refine_strategy,
                                  &error_integrator);*/
AmandusIntegrator<d>* AmandInt = 0;
//Function<d>* startup = 0;

global_refinement_linear_loop(param.get_integer("MaxDofs"),
				app,
				solver,
				residual,
				AmandInt,
				AmandInt);
				//startup, false);
}
