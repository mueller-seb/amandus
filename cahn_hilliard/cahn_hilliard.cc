#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_tools.h>

#include <apps.h>

#include <cahn_hilliard/residual.h>
#include <cahn_hilliard/matrix.h>
#include <cahn_hilliard/samples.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;


int main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(2);

  AmandusParameters param;

  param.enter_subsection("Model");
  param.declare_entry("Startup", "1");
  param.leave_subsection();

  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Model");
  int startup_no = param.get_integer("Startup");
  param.leave_subsection();

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d> > fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr;
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  std::vector<bool> mask;
  mask.push_back(false);
  mask.push_back(true);
  BlockMask timemask(mask);

  CahnHilliard::Matrix<d> matrix_stationary;
  Integrators::Theta<d> matrix_integrator(matrix_stationary,
                                          true, timemask);
  matrix_integrator.input_vector_names.push_back("Newton iterate");
  CahnHilliard::Residual<d> residual_integrator;
  Integrators::Theta<d> explicit_integrator(residual_integrator,
                                            false, timemask, true);
  explicit_integrator.input_vector_names.push_back("Previous iterate");
  Integrators::Theta<d> implicit_integrator(residual_integrator,
                                            true, timemask);
  implicit_integrator.input_vector_names.push_back("Newton iterate");

  AmandusApplicationSparse<d> app(tr, *fe, true);
  app.parse_parameters(param);
  //AmandusApplication<d> app(tr, *fe);
  //app.set_meanvalue();
  AmandusResidual<d> expl(app, explicit_integrator);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, implicit_integrator);

  // Set up timestepping algorithm with embedded Newton solver

  param.enter_subsection("Output");
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.parse_parameters(param);
  newout.initialize(app.dofs());
  param.leave_subsection();

  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.parse_parameters(param);
  newton.debug = 6;

  Algorithms::ThetaTimestepping<Vector<double> > timestepping(expl, newton);
  timestepping.set_output(newout);
  timestepping.parse_parameters(param);

  // Now we prepare for the actual timestepping

  timestepping.notify(dealii::Algorithms::Events::remesh);
  dealii::Vector<double> solution;
  app.setup_system();
  app.setup_vector(solution);

  Function<d>& startup = *CahnHilliard::selector(startup_no);
  VectorTools::interpolate(app.dofs(), startup, solution);

  dealii::AnyData indata;
  indata.add(&solution, "solution");
  dealii::AnyData outdata;
  timestepping(indata, outdata);
}
