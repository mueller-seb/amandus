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
#include <deal.II/fe/fe_tools.h>

#include <debug/visualize_solution.h>


int main(int argc, const char** argv)
{
  using namespace dealii;

  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  AmandusParameters param;

  param.enter_subsection("CheckerboardPattern");
  param.declare_entry("Quadrant1", "1.0", Patterns::Double());
  param.declare_entry("Quadrant2", "1.0", Patterns::Double());
  param.declare_entry("Quadrant3", "1.0", Patterns::Double());
  param.declare_entry("Quadrant4", "1.0", Patterns::Double());
  param.leave_subsection();
  param.enter_subsection("AmandusApplication");
  param.declare_entry("Multigrid", "false", Patterns::Bool());
  param.declare_entry("Steps", "1", Patterns::Integer());
  param.leave_subsection();

  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  const FiniteElement<d>* fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr;
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  param.enter_subsection("CheckerboardPattern");
  std::vector<double> coefficient_parameters;
  coefficient_parameters.push_back(param.get_double("Quadrant1"));
  coefficient_parameters.push_back(param.get_double("Quadrant2"));
  coefficient_parameters.push_back(param.get_double("Quadrant3"));
  coefficient_parameters.push_back(param.get_double("Quadrant4"));
  param.leave_subsection();

  // a diffusion tensor that is piecewise constant in the quadrants
  Darcy::Checkerboard::CheckerboardTensorFunction
    d_tensor(coefficient_parameters);
  // divergence free mixed solution corresponding to the diffusion tensor
  Darcy::Checkerboard::MixedSolution mixed_solution(coefficient_parameters);

  // System integrator for mixed discretization of Darcy's equation for
  // given weight tensor
  Darcy::SystemIntegrator<2> system_integrator(d_tensor.inverse());
  // Right hand side corresponding to the exact solution's boundary values
  // and zero divergence
  Darcy::RHSIntegrator<2> rhs_integrator(mixed_solution.scalar_solution);
  // Error integrator, use d_tensor weighted L2 inner product for velocity
  Darcy::ErrorIntegrator<2> error(mixed_solution, d_tensor);

  param.enter_subsection("AmandusApplication");
  AmandusApplicationSparse<d>* app;
  unsigned int steps = param.get_integer("Steps");
  // construct both, otherwise we run into problems with subscriptions...
  AmandusApplication<d> amandus_mg(tr, *fe);
  AmandusApplicationSparse<d> amandus_umf(tr, *fe, true);
  if(param.get_bool("Multigrid"))
  {
    //app = new AmandusApplication<d>(tr, *fe);
    app = &amandus_mg;
  } else {
    //app = new AmandusApplicationSparse<d>(tr, *fe, true);
    app = &amandus_umf;
  }
  param.leave_subsection();

  AmandusSolve<d> solver(*app, system_integrator);
  AmandusResidual<d> residual(*app, rhs_integrator);

  app->parse_parameters(param);

  global_refinement_linear_loop(steps, *app, solver, residual, &error);

  // output of exact mixed solution for comparison
  param.enter_subsection("Output");
  QGauss<d> quadrature(fe->tensor_degree() + 2);
  debug::output_solution(mixed_solution,
                         app->dofs(),
                         quadrature,
                         param);
  param.leave_subsection();

  //solver.~AmandusSolve<d>();
  //residual.~AmandusResidual<d>();
  //delete app;
  //delete fe;

  return 0;
}
