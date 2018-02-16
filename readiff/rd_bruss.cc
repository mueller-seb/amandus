// $Id$

/**
 * @file
 *
 * @brief Brusselator problem as instationary ReactionDiffusion problem
 * <ul>
 * <li>Instationary ReactionDiffusion model</li>
 * <li>Homogeneous Neumann boundary conditions</li>
 * <li>Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * Parameter set and initial condition can be found in
 * Hairer, Norsett, Wanner: Solving Ordinary Differential
 * Equations I, 2000.
 *
 * @ingroup ReaDiffgroup
 * @ingroup Brusselatorgroup
 *
 */

#include <amandus/apps.h>
#include <amandus/readiff/matrix.h>
#include <amandus/readiff/residual.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <boost/scoped_ptr.hpp>

template <int dim>
class Startup : public dealii::Function<dim>
{
public:
  Startup();
  virtual void vector_value_list(const std::vector<Point<dim>>& points,
                                 std::vector<Vector<double>>& values) const;
};

template <int dim>
Startup<dim>::Startup()
  : Function<dim>(2)
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

    values[k](0) = .5 + p(1);
    values[k](1) = 1. + 5. * p(0);
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
  ReactionDiffusion::Parameters::declare_parameters(param);
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  std::unique_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  ReactionDiffusion::Parameters parameters;
  parameters.parse_parameters(param);

  ReactionDiffusion::Matrix<d> m_integrator(parameters);
  Integrators::Theta<d> matrix_integrator(m_integrator, true);
  matrix_integrator.input_vector_names.push_back("Newton iterate");
  ReactionDiffusion::Residual<d> r_integrator(parameters);
  Integrators::Theta<d> implicit_integrator(r_integrator, true);
  implicit_integrator.input_vector_names.push_back("Newton iterate");
  Integrators::Theta<d> explicit_integrator(r_integrator, false);
  explicit_integrator.input_vector_names.push_back("Previous iterate");

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
