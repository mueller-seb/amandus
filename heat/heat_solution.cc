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
#include <amandus/heat/matrix_heat.h>
//#include <amandus/heat/rhs_one.h>
#include <amandus/heat/RHS_heat.h>
#include <amandus/heat/heat_solution.h>
#include <amandus/refine_strategy.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>



#include <boost/scoped_ptr.hpp>

class Solution : public Function<2>
{
public:
  Solution(const double eps = 1e-5);

  virtual double value(const Point<2>& p, const unsigned int component = 0) const;

  virtual void value_list(const std::vector<Point<2>>& points, std::vector<double>& values,
                          const unsigned int component = 0) const;
  virtual Tensor<1, 2> gradient(const Point<2>& p, const unsigned int component = 0) const;

  virtual double laplacian(const Point<2>& p, const unsigned int component = 0) const;

private:
  const double eps;
};

Solution::Solution(const double eps)
  : eps(eps)
{
}

double
Solution::value(const Point<2>& p, const unsigned int) const
{
  const double x = p[0];
  const double y = p[1];

  return (x*x-1)*(y*y-1);
}

void
Solution::value_list(const std::vector<Point<2>>& points, std::vector<double>& values,
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
Solution::laplacian(const Point<2>& p, const unsigned int) const
{
  const double x = p[0];
  const double y = p[1];

  double val = 2*(y*y-1)+2*(x*x-1);
  /*if (abs(y) < eps)
     val = val - 2;*/
  return val;
}

Tensor<1, 2>
Solution::gradient(const Point<2>& p, const unsigned int) const
{
  Tensor<1, 2> val;
  const double x = p[0];
  const double y = p[1];

  val[0] = 2*x*(y*y-1);
  val[1] = 2*y*(x*x-1);
  /*if (abs(y) < eps)
    val[0] = val[0] - 2*x;*/
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
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();


  HeatIntegrators::MatrixHeat<d> matrix_integrator;

  Solution exact_solution;

  HeatIntegrators::SolutionRHS<d> rhs_integrator(exact_solution);
  HeatIntegrators::SolutionError<d> error_integrator(exact_solution);
  HeatIntegrators::SolutionEstimate<d> estimate_integrator(exact_solution);


  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  RefineStrategy::MarkBulk<d> refine_strategy(tr, 0.5);

  adaptive_refinement_linear_loop(param.get_integer("MaxDofs"),
                                  app,
                                  tr,
                                  solver,
                                  residual,
                                  estimate_integrator,
                                  refine_strategy,
				  &error_integrator);
}
