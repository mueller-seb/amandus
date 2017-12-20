/**
 * @file
 * <ul>
 * <li> Instationary Biot equations</li>
 * <li> smooth version Barry/Mercer example</li>
 * <li> Newton solver</li>
 * </ul>
 *
 * @ingroup Examples
 */

#include <amandus/apps.h>
#include <amandus/biot/barry_mercer.h>
#include <biot/parameters.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(3);

  AmandusParameters param;
  ::Biot::Parameters::declare_parameters(param);
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, 0, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  deallog << "Mesh: " << tr.n_levels() << " levels, " << tr.n_active_cells() << " cells"
          << std::endl;

  ZeroFunction<d> startup(2 * d + 1);

  ::Biot::Parameters parameters;
  parameters.parse_parameters(param);
  ::Biot::BarryMercerMatrix<d> matrix_integrator(parameters);
  ::Biot::BarryMercerSource<d> explicit_integrator(parameters, false);
  explicit_integrator.input_vector_names.push_back("Previous iterate");
  ::Biot::BarryMercerSource<d> implicit_integrator(parameters, true);
  implicit_integrator.input_vector_names.push_back("Newton iterate");
  ::Biot::BarryMercerError<d> error_integrator(parameters, .5);

  AmandusUMFPACK<d> app(tr, *fe);
  app.parse_parameters(param);

  AmandusResidual<d> expl(app, explicit_integrator);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, implicit_integrator);

  param.enter_subsection("Output");
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.parse_parameters(param);
  newout.initialize(app.dofs());
  param.leave_subsection();

  Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.parse_parameters(param);

  Algorithms::ThetaTimestepping<Vector<double>> timestepping(expl, newton);
  //  timestepping.set_output(newout);
  timestepping.parse_parameters(param);

  // Now we prepare for the actual timestepping

  timestepping.notify(dealii::Algorithms::Events::remesh);
  dealii::Vector<double> solution;
  app.setup_system();
  app.setup_vector(solution);

  dealii::QGauss<d> quadrature(app.dofs().get_fe().tensor_degree() + 1);
  dealii::VectorTools::project(app.dofs(), app.hanging_nodes(), quadrature, startup, solution);

  dealii::AnyData indata;
  indata.add(&solution, "solution");
  dealii::AnyData outdata;
  timestepping(indata, outdata);

  app.output_results(tr.n_levels(), &indata);

  dealii::BlockVector<double> errors;
  errors.reinit(error_integrator.n_errors());
  app.error(errors, indata, error_integrator);
  for (unsigned int i = 0; i < errors.n_blocks(); ++i)
  {
    dealii::deallog << "Error(";
    if (error_integrator.error_name(i) != std::string())
      dealii::deallog << error_integrator.error_name(i);
    else
      dealii::deallog << i;
    dealii::deallog << "): " << errors.block(i).l2_norm() << std::endl;
  }
}
