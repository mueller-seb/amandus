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
#include <amandus/heat/rhs.h>
#include <amandus/refine_strategy.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <boost/scoped_ptr.hpp>

#include <amandus/heat/conductivity.h>

template <int dim>
class Force : public Function<dim>
{
public:
  Force(const double margin = 0.0);
  virtual double value(const Point<dim>& p, const unsigned int component) const override;
  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component) const override;
  private:
    const double margin; //MARGIN between low dimensional embedding/pole and boundaries
};

template <int dim>
Force<dim>::Force(const double margin) : margin(margin)
{
}

template <int dim>
double
Force<dim>::value(const Point<dim>& p, const unsigned int component) const
{
  double x = p(0);
  double y = p(1);
  bool onEmbedding = (abs(y) < 1e-5) && (abs(x) <= (1-margin));

  double result = 0; //on face, but not on embedding
  if ((component == 0) || onEmbedding)
  { //beyond faces or on embedding
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
                         const unsigned int component) const
{
  AssertDimension(points.size(), values.size());

  for (unsigned int k = 0; k < points.size(); ++k)
  {
    const Point<dim>& p = points[k];
    values[k] = value(p, component);
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
  Force<d> f(margin);
  Conductivity<d> kappa(margin);

  HeatIntegrators::Matrix<d> matrix_integrator(kappa);
  HeatIntegrators::RHS<d> rhs_integrator(f);
  HeatIntegrators::Estimate<d> estimate_integrator(f);
  AmandusIntegrator<d>* error_integrator = 0;

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
                                  error_integrator);
}
