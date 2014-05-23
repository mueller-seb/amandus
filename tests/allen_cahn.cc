// $Id$

/**
 * @file
 *
 * @brief Instationary Allen-Cahn equations
 * <ul>
 * <li>Instationary Allen-Cahn equations</li>
 * <li>Homogeneous Dirichlet boundary conditions</li>
 * <li>Exact polynomial solutionExact polynomial solution</li>
 * <li>Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <deal.II/base/function_lib.h>
#include <apps.h>
#include <allen_cahn/implicit.h>
#include <allen_cahn/explicit.h>
#include <allen_cahn/matrix.h>


int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
//  deallog.depth_console(2);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -2, 2);
  tr.refine_global(6);
  
  const unsigned int degree = 1;
  FE_DGQ<d> fe(degree);

  const double diffusion = 1.e-2;
  AllenCahn::Matrix<d> matrix_integrator(diffusion);
  AllenCahn::ExplicitResidual<d> explicit_integrator(diffusion);
  explicit_integrator.input_vector_names.push_back("Previous iterate");
  AllenCahn::ImplicitResidual<d> implicit_integrator(diffusion);
  implicit_integrator.input_vector_names.push_back("Newton iterate");

  AmandusApplicationSparseMultigrid<d> app(tr, fe);
  AmandusResidual<d> expl(app, explicit_integrator);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, implicit_integrator);

  // Set up timestepping algorithm with embedded Newton solver
  
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dof_handler);
  
  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.control.log_history(true);
  newton.control.set_reduction(1.e-14);
  newton.threshold(.2);
  
  Algorithms::ThetaTimestepping<Vector<double> > timestepping(expl, newton);
  timestepping.set_output(newout);
  timestepping.theta(.55);
  timestepping.timestep_control().start_step(.1);
  timestepping.timestep_control().final(10.);

  // Now we prepare for the actual timestepping
  
  timestepping.notify(dealii::Algorithms::Events::remesh);
  dealii::Vector<double> solution;
  app.setup_system();
  app.setup_vector(solution);
  
  Functions::CosineFunction<d> cosine;
  VectorTools::interpolate(app.dof_handler, cosine, solution);
  
  dealii::AnyData indata;
  indata.add(&solution, "solution");
  dealii::AnyData outdata;
  timestepping(indata, outdata);  
}
