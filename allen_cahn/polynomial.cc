/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 *
 * @brief Stationary Allen-Cahn with manufactured solution
 * <ul>
 * <li>Stationary Allen-Cahn equations</li>
 * <li>Homogeneous Dirichlet boundary conditions</li>
 * <li>Exact polynomial solution</li>
 * <li>Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Allen_Cahngroup
 */
#include <deal.II/algorithms/newton.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>

#include <amandus/allen_cahn/matrix.h>
#include <amandus/allen_cahn/polynomial.h>
#include <amandus/allen_cahn/residual.h>
#include <amandus/apps.h>
#include <amandus/tests.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;

template <int d>
void
run(AmandusParameters& param)
{
  param.enter_subsection("Discretization");
  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));

  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));
  param.leave_subsection();

  Polynomials::Polynomial<double> solution1d;
  // solution1d += Polynomials::Monomial<double>(3, -1.);
  solution1d += Polynomials::Monomial<double>(2, -1.);
  solution1d += Polynomials::Monomial<double>(1, 3.);
  solution1d.print(std::cout);
  TensorProductPolynomial<d> exact_solution(solution1d);

  param.enter_subsection("Model");
  AllenCahn::Matrix<d> matrix_integrator(param.get_double("Diffusion"));
  matrix_integrator.input_vector_names.push_back("Newton iterate");
  AllenCahn::Residual<d> residual_integrator(param.get_double("Diffusion"));
  residual_integrator.input_vector_names.push_back("Newton iterate");
  param.leave_subsection();

  Integrators::L2ErrorIntegrator<d> l2_error_integrator;
  Integrators::H1ErrorIntegrator<d> h1_error_integrator;
  ErrorIntegrator<d> error_integrator(exact_solution);
  error_integrator.add(&l2_error_integrator);
  error_integrator.add(&h1_error_integrator);

  AmandusApplicationSparse<d>* app_init;
  param.enter_subsection("Testing");
  if (param.get_bool("Multigrid"))
  {
    app_init = new AmandusApplication<d>(tr, *fe);
  }
  else
  {
    app_init = new AmandusApplicationSparse<d>(tr, *fe, param.get_bool("UMFPack"));
  }
  param.leave_subsection();
  boost::scoped_ptr<AmandusApplicationSparse<d>> app(app_init);
  app->parse_parameters(param);
  AmandusSolve<d> solver(*app, matrix_integrator);
  ExactResidual<d> residual(*app, residual_integrator, exact_solution, 4);

  param.enter_subsection("Output");
  Algorithms::DoFOutputOperator<Vector<double>, d> newton_output;
  newton_output.initialize(app->dofs());
  newton_output.parse_parameters(param);
  param.leave_subsection();
  Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.initialize(newton_output);
  newton.parse_parameters(param);

  param.enter_subsection("Testing");
  int steps = param.get_integer("Number of global refinement loops");
  double TOL = param.get_double("Tolerance");
  param.leave_subsection();
  BlockVector<double> errors(2);
  double acc_error;
  for (int s = 0; s < steps; ++s)
  {
    iterative_solve_and_error<d>(errors, *app, newton, error_integrator);
    for (unsigned int i = 0; i < errors.n_blocks(); ++i)
    {
      acc_error = errors.block(i).l2_norm();
      deallog << "Error(" << i << "): " << acc_error << std::endl;
      Assert(acc_error < TOL, ExcErrorTooLarge(acc_error));
    }
    tr.refine_global(1);
  }
}

int
main(int argc, const char** argv)
{
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  AmandusParameters param;

  param.enter_subsection("Testing");
  param.declare_entry("Number of global refinement loops", "3");
  param.declare_entry("Tolerance", "1.e-13");
  param.declare_entry("Multigrid", "true");
  param.declare_entry("UMFPack", "true");
  param.leave_subsection();

  param.enter_subsection("Model");
  param.declare_entry("Diffusion", "1.0");
  param.declare_entry("Dimensionality", "2");
  param.leave_subsection();

  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Model");
  const unsigned int d = param.get_integer("Dimensionality");
  param.leave_subsection();

  if (d == 3)
  {
    run<3>(param);
  }
  else
  {
    run<2>(param);
  }
}
