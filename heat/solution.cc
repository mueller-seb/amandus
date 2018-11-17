/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * \brief Example for Laplacian with manufactured solution on adaptive meshes
 * <ul>
 * <li> Stationary Poisson equations</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Exact solution</li>
 * <li> Error computation</li>
 * <li> Error estimation</li>
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

#include <boost/scoped_ptr.hpp>

#include <amandus/heat/conductivity.h>
#include <amandus/heat/u.h>

template <int dim>
class RHSWrapper : public Function<dim>
{
public:
  RHSWrapper(Solution<dim>& u);
  virtual double value(const Point<dim>& p, const unsigned int component) const override;
  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component) const override;
private:
  SmartPointer<Solution<dim>, RHSWrapper<dim>> u;
};

template <int dim>
RHSWrapper<dim>::RHSWrapper(Solution<dim>& u) : u(&u)
{
}

template <int dim>
double
RHSWrapper<dim>::value(const Point<dim>& p, const unsigned int component) const
{
  return -u->laplacian(p, component);
}

template <int dim>
void
RHSWrapper<dim>::value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                         const unsigned int component) const
{
  AssertDimension(points.size(), values.size());

  for (unsigned int k = 0; k < points.size(); ++k)
  {
    const Point<dim>& p = points[k];
    values[k] = -u->laplacian(p, component);
  }
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
  std::unique_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  const double margin = 0.0;

  Conductivity<d> kappa(margin);
  Solution<d> exact_solution(kappa);
  RHSWrapper<d> f(exact_solution);

  HeatIntegrators::Matrix<d> matrix_integrator(kappa);
  HeatIntegrators::SolutionRHS<d> rhs_integrator(exact_solution);
  HeatIntegrators::SolutionError<d> error_integrator(exact_solution);

  HeatIntegrators::SolutionEstimate<d> estimate_integrator(exact_solution);

  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  RefineStrategy::MarkBulk<d> refine_strategy(tr, 0.5);
  //RefineStrategy::MarkUniform<d> refine_strategy(tr);
  adaptive_refinement_linear_loop(param.get_integer("MaxDofs"),
                                  app,
                                  tr,
                                  solver,
                                  residual,
                                  estimate_integrator,
                                  refine_strategy,
				  &error_integrator);
}
