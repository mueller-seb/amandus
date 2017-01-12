/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * <ul>
 * <li> Stationary advection equations</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Linear solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Advectiongroup
 */

#include <advection/parameters.h>
#include <amandus/advection/matrix.h>
#include <amandus/advection/polynomial.h>
#include <amandus/tests.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <boost/scoped_ptr.hpp>

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  AmandusParameters param;
  ::Advection::Parameters::declare_parameters(param);
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(2, -1.);
  // solution1d += Polynomials::Monomial<double>(0, 1.);
  solution1d.print(std::cout);

  std::vector<Polynomials::Polynomial<double>> potentials(1);
  potentials[0] = solution1d;

  ::Advection::Parameters parameters;
  parameters.parse_parameters(param);
  ::Advection::Matrix<d> matrix_integrator(parameters);
  ::Advection::PolynomialRHS<d> rhs_integrator(parameters, potentials);
  ::Advection::PolynomialError<d> error_integrator(parameters, potentials);

  AmandusUMFPACK<d> app(tr, *fe);
  app.parse_parameters(param);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);

  BlockVector<double> errors(2);
  Vector<double> acc_errors(2);
  solve_and_error(errors, app, solver, residual, error_integrator);
  for (unsigned int i = 0; i < errors.n_blocks(); ++i)
  {
    acc_errors(i) = errors.block(i).l2_norm();
    deallog << "Error(" << i << "): " << acc_errors(i) << std::endl;
  }
  Assert(acc_errors(0) < 1.e-14, ExcErrorTooLarge(acc_errors(0)));
  Assert(acc_errors(1) < 1.e-13, ExcErrorTooLarge(acc_errors(1)));
}
