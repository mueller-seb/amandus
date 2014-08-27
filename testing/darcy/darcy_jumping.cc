/**
 * @file
 *
 * Exact solution to divergence free Darcy problem with jumping
 * coefficients.
 *
 * @ingroup Examples
 */

#include <apps.h>
#include <darcy/checkerboard/solution.h>
#include <darcy/integrators.h>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <debug/visualize_solution.h>

int main(int argc, const char** argv)
{
  using namespace dealii;

  const unsigned int d = 2;

  const int initial_refinement = 3;
  const unsigned int steps = 1;
  const int degree = 1;
  const int quadrature_degree = degree + 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  AmandusParameters param;
  param.read(argc, argv);
  param.log_parameters(deallog);

  Triangulation<d> tr;
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(initial_refinement);

  FE_RaviartThomas<d> vec(degree);
  FE_DGQ<d> scal(degree);
  FESystem<d> fe(vec, 1, scal, 1);

  std::vector<double> coefficient_parameters;
  coefficient_parameters.push_back(10.0);
  coefficient_parameters.push_back(1.0);
  coefficient_parameters.push_back(10.0);
  coefficient_parameters.push_back(1.0);

  // a diffusion tensor that is piecewise constant in the quadrants
  Darcy::Checkerboard::CheckerboardTensorFunction<2>
    d_tensor(coefficient_parameters);
  // divergence free mixed solution corresponding to the diffusion tensor
  Darcy::Checkerboard::MixedSolution<2> mixed_solution(coefficient_parameters);

  // System integrator for mixed discretization of Darcy's equation for
  // given weight tensor
  Darcy::SystemIntegrator<2> system_integrator(d_tensor.inverse());
  // Right hand side corresponding to the exact solution's boundary values
  // and zero divergence
  Darcy::RHSIntegrator<2> rhs_integrator(mixed_solution.scalar_solution);


  /*
  DarcyCoefficient::Parameters<d> problem_parameters(coefficient_parameters);

  DarcyCoefficient::SystemIntegrator<d> system_integrator(problem_parameters);
  DarcyCoefficient::RHSIntegrator<d> rhs_integrator(problem_parameters);
  */

  AmandusApplicationSparse<d> app(tr, fe, true);
  AmandusSolve<d> solver(app, system_integrator);
  AmandusResidual<d> residual(app, rhs_integrator); // really just rhs

  app.parse_parameters(param);

  app.control.set_reduction(1.e-10);
  app.control.set_max_steps(50000);

  global_refinement_linear_loop(steps, app, solver, residual);

  param.enter_subsection("Output");
  QGauss<d> quadrature(quadrature_degree);
  debug::output_solution(mixed_solution,
                         app.dofs(),
                         quadrature,
                         param);
  param.leave_subsection();

  return 0;
}
