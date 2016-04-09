/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * <ul>
 * <li> Stationary Darcy-Stokes equations</li>
 * <li> Homogeneous no-slip boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Linear solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 */

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <amandus/apps.h>
#include <amandus/brinkman/matrix.h>
#include <amandus/rhs_one.h>


int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  std::vector<unsigned int> sub(d,3);
  Point<d> p1;
  Point<d> p2;

  for (unsigned int i=0;i<d;++i)
    {
      p1(i) = -1.;
      p2(i) = 1.;
    }
  
  GridGenerator::subdivided_hyper_rectangle (tr, sub, p1, p2);

  const unsigned int degree = 1;
  FE_RaviartThomas<d> vec(degree);
  FE_DGQ<d> scal(degree);
  FESystem<d> fe(vec, 1, scal, 1);

  Brinkman::Parameters coefficients(1000);
  Brinkman::Matrix<d> matrix_integrator(coefficients);
  RhsOne<d> rhs_integrator;

  AmandusApplicationSparseMultigrid<d> app(tr, fe);
  app.set_boundary(0);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d>    residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);
  
  global_refinement_linear_loop(5, app, solver, residual);
}
