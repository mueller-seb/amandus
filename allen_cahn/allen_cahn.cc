/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 *
 * @brief Instationary Allen-Cahn equations
 * <ul>
 * <li>Instationary Allen-Cahn equations</li>
 * <li>Homogeneous Neumann boundary conditions</li>
 * <li>Exact polynomial solutionExact polynomial solution</li>
 * <li>Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Allen_Cahngroup
 */

#include <amandus/allen_cahn/matrix.h>
#include <amandus/allen_cahn/residual.h>
#include <amandus/apps.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;

template <int dim>
class Startup : public dealii::Function<dim>
{
public:
  Startup();
  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component = 0) const override;
  virtual void vector_value_list(const std::vector<Point<dim>>& points,
                                 std::vector<Vector<double>>& values) const override;
};

template <int dim>
Startup<dim>::Startup()
  : Function<dim>(1)
{
}

template <int dim>
void
Startup<dim>::vector_value_list(const std::vector<Point<dim>>& points,
                                std::vector<Vector<double>>& values) const
{
  AssertDimension(points.size(), values.size());

  for (unsigned int k = 0; k < points.size(); ++k)
  {
    const Point<dim>& p = points[k];
    if (std::fabs(p(0)) < .8 && std::fabs(p(1)) < .2)
      values[k](0) = 1.;
    else if (std::fabs(p(1)) < .8 && std::fabs(p(0)) < .2)
      values[k](0) = 1.;
    else
      values[k](0) = -1.;
  }
}

template <int dim>
void
Startup<dim>::value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                         const unsigned int) const
{
  AssertDimension(points.size(), values.size());

  for (unsigned int k = 0; k < points.size(); ++k)
  {
    const Point<dim>& p = points[k];
    if (std::fabs(p(0)) < .8 && std::fabs(p(1)) < .2)
      values[k] = 1.;
    else if (std::fabs(p(1)) < .8 && std::fabs(p(0)) < .2)
      values[k] = 1.;
    else
      values[k] = -1.;
  }
}

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  AmandusParameters param;
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  std::unique_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  const double diffusion = 2.e-3;
  AllenCahn::Matrix<d> matrix_stationary(diffusion);
  Integrators::Theta<d> matrix_integrator(matrix_stationary, true);
  matrix_integrator.input_vector_names.push_back("Newton iterate");
  AllenCahn::Residual<d> residual_integrator(diffusion);
  Integrators::Theta<d> explicit_integrator(residual_integrator, false);
  explicit_integrator.input_vector_names.push_back("Previous iterate");
  Integrators::Theta<d> implicit_integrator(residual_integrator, true);
  implicit_integrator.input_vector_names.push_back("Newton iterate");

  AmandusApplication<d> app(tr, *fe);
  AmandusResidual<d> expl(app, explicit_integrator);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, implicit_integrator);

  // Set up timestepping algorithm with embedded Newton solver

  param.enter_subsection("Output");
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.parse_parameters(param);
  newout.initialize(app.dofs());
  param.leave_subsection();

  Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.parse_parameters(param);

  Algorithms::ThetaTimestepping<Vector<double>> timestepping(expl, newton);
  timestepping.set_output(newout);
  timestepping.parse_parameters(param);

  // Now we prepare for the actual timestepping

  timestepping.notify(dealii::Algorithms::Events::remesh);
  dealii::Vector<double> solution;
  app.setup_system();
  app.setup_vector(solution);

  Startup<d> startup;
  VectorTools::interpolate(app.dofs(), startup, solution);

  dealii::AnyData indata;
  indata.add(&solution, "solution");
  dealii::AnyData outdata;
  timestepping(indata, outdata);
}
