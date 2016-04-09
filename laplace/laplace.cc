/**********************************************************************
 *  Copyright (C) 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 *
 * @brief Laplace
 * <ul>
 * <li> Stationary laplace equation</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Linear solver</li>
 * </ul>
 *
 * @author Anja Bettendorf
 *
 * @ingroup Examples
 */

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <amandus/apps.h>
#include <amandus/laplace/matrix.h>
#include <amandus/laplace/matrix_factor.h>
#include <amandus/laplace/noforce.h>
#include <amandus/laplace/polynomial.h>

int main()
{
  const unsigned int d=2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);

  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1);
  tr.refine_global(3);

  const unsigned int degree = 2;
  FE_DGQ<d> fe(degree);

  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(3, -1.);
  solution1d += Polynomials::Monomial<double>(1, 3.);
  solution1d.print(std::cout);

  // without factor
  //LaplaceIntegrators::Matrix<d> matrix_integrator;

  // with factor
  double fakt=1.;
  LaplaceIntegrators::MatrixFaktor<d> matrix_integrator(fakt);

  LaplaceIntegrators::PolynomialRHS<d> rhs(solution1d);
  LaplaceIntegrators::PolynomialError<d> error(solution1d);

  AmandusUMFPACK<d>  app(tr, fe);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs);


  global_refinement_linear_loop(3, app, solver, residual, &error);

}
