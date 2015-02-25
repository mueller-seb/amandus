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
 * @ingroup Examples
 */
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>

#include <apps.h>
#include <tests.h>
#include <allen_cahn/polynomial.h>
#include <allen_cahn/residual.h>
#include <allen_cahn/matrix.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;

int main(int argc, const char** argv)
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);

  AmandusParameters param;

  param.enter_subsection("Testing");
  param.declare_entry("Number of global refinement loops", "3");
  param.declare_entry("Tolerance", "1.e-13");
  param.leave_subsection();

  param.read(argc, argv);
  param.log_parameters(deallog);
  
  param.enter_subsection("Discretization");
  Triangulation<d> tr;
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  
  boost::scoped_ptr<const FiniteElement<d> >
    fe(FETools::get_fe_by_name<d, d>(param.get("FE")));
  param.leave_subsection();

  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(3, -1.);
  solution1d += Polynomials::Monomial<double>(1, 3.);
  solution1d.print(std::cout);
  TensorProductPolynomial<d> exact_solution(solution1d);
  
  AllenCahn::Matrix<d> matrix_integrator(1.);
  matrix_integrator.input_vector_names.push_back("Newton iterate");
  AllenCahn::Residual<d> residual_integrator(1.);
  residual_integrator.input_vector_names.push_back("Newton iterate");
  //AllenCahn::PolynomialResidual<d> residual_integrator(1., solution1d);
  AllenCahn::PolynomialError<d> error_integrator(solution1d);
  //ExactProjectionError<d> error_integrator;
  
  AmandusApplicationSparseMultigrid<d> app(tr, *fe);
  //AmandusApplicationSparse<d> app(tr, *fe);
  AmandusSolve<d>       solver(app, matrix_integrator);
  ExactResidual<d> residual(app, residual_integrator);
  
  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.parse_parameters(param);
  
  param.enter_subsection("Testing");
  int steps = param.get_integer("Number of global refinement loops");
  double TOL = param.get_double("Tolerance");
  param.leave_subsection();
  BlockVector<double> errors(2);
  double acc_error;
  for(int s = 0; s < steps; ++s)
  {
    iterative_solve_and_error<d>(
        errors,
        app,
        newton,
        error_integrator,
        0,
        &exact_solution);
    for(unsigned int i=0;i<errors.n_blocks();++i)
    {
      acc_error = errors.block(i).l2_norm();
      deallog << "Error(" << i << "): " << acc_error << std::endl;
      Assert(acc_error < TOL, ExcErrorTooLarge(acc_error));
    }
    tr.refine_global(1);
  }
  //global_refinement_nonlinear_loop(steps, app, newton, &error_integrator);
}
