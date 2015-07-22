/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 *
 * @brief Instationary Brusselator problem
 * <ul>
 * <li>Instationary Brusselator model</li>
 * <li>Homogeneous Dirichlet boundary conditions</li>
 * <li>Exact polynomial solutionExact polynomial solution</li>
 * <li>Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */
#include <deal.II/base/logstream.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/base/function_lib.h>
#include <brusselator/ode.h>

#include <fstream>

int main()
{
  std::ofstream logfile("deallog");
  dealii::deallog.attach(logfile);
  dealii::deallog.depth_console(2);
  
  Brusselator::Parameters parameters;
  parameters.alpha0 = .002;
  parameters.alpha1 = .002;
  parameters.A = 3.4;
  parameters.B = 1.;

  Explicit         expl(parameters);
  ImplicitSolve    solver(parameters);
  ImplicitResidual residual(parameters);

  // Set up timestepping algorithm with embedded Newton solver

  std::ofstream os("ode.out");
  dealii::Algorithms::OutputOperator<dealii::Vector<double> > newout;
  newout.initialize_stream(os);
  
  dealii::Algorithms::Newton<dealii::Vector<double> > newton(residual, solver);
  newton.control.log_history(true);
  newton.control.set_reduction(1.e-14);
  newton.threshold(.2);
  
  dealii::Algorithms::ThetaTimestepping<dealii::Vector<double> > timestepping(expl, newton);
  timestepping.set_output(newout);
  timestepping.theta(.5);
  timestepping.timestep_control().start_step(.1);
  timestepping.timestep_control().final(20.);

  // Now we prepare for the actual timestepping
  
  timestepping.notify(dealii::Algorithms::Events::remesh);
  dealii::Vector<double> solution(2);
  solution(0) = 2.;
  solution(1) = 1.;
  
  dealii::AnyData indata;
  indata.add(&solution, "solution");
  dealii::AnyData outdata;
  timestepping(indata, outdata);  
}
