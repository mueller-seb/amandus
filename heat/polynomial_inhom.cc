/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * <ul>
 * <li> Stationary Poisson equations</li>
 * <li> Inhomogeneous Dirichlet boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Non-linear solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @author Joscha Gedicke
 *
 * @ingroup Laplacegroup
 */

#include <amandus/apps.h>
#include <amandus/laplace/matrix.h>
#include <amandus/laplace/polynomial.h>
#include <deal.II/algorithms/newton.h>
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
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  std::unique_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(4, 1.);
  solution1d += Polynomials::Monomial<double>(2, -2.);
  solution1d += Polynomials::Monomial<double>(0, 1.);
  solution1d += Polynomials::Monomial<double>(0, 1.);
  solution1d.print(std::cout);

  LaplaceIntegrators::Matrix<d> matrix_integrator;
  LaplaceIntegrators::PolynomialResidual<d> rhs_integrator(solution1d);
  rhs_integrator.input_vector_names.push_back("Newton iterate");
  LaplaceIntegrators::PolynomialError<d> error_integrator(solution1d);
  LaplaceIntegrators::PolynomialBoundary<d> bd_function(solution1d);

  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);

  Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.parse_parameters(param);

  global_refinement_nonlinear_loop(5,
                                   app,
                                   newton,
                                   &error_integrator,
                                   static_cast<AmandusIntegrator<d>*>(nullptr),
                                   static_cast<dealii::Function<d>*>(nullptr),
                                   &bd_function);
}
