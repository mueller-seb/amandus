/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __tests_h
#define __tests_h

#include <deal.II/base/logstream.h>
#include <deal.II/algorithms/any_data.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>

#include <amandus.h>

/**
 * Error is not close to machine accuracy
 */
DeclException1(ExcErrorTooLarge, double,
	       << "error " << arg1 << " is too large");

/**
 * This function solves an equation on a given mesh and computes the
 * errors.
 *
 * The errors per cell are returned in the BlockVector used as first
 * argument. The number of blocks has to be at least the number of
 * errors computed by the error integrator, but the blocks may be
 * empty. They will be resized.
 *
 * @ingroup Verification
 */
template <int dim>
void
solve_and_error(dealii::BlockVector<double>& errors,
		AmandusApplicationSparse<dim> &app,
		dealii::Algorithms::Operator<dealii::Vector<double> >& solver,
		dealii::Algorithms::Operator<dealii::Vector<double> >& residual,
		const AmandusIntegrator<dim>& error)
{
  dealii::Vector<double> res;
  dealii::Vector<double> sol;
  
  solver.notify(dealii::Algorithms::Events::remesh);
  app.setup_system();
  app.setup_vector(res);
  app.setup_vector(sol);
  
  dealii::AnyData solution_data;
  dealii::Vector<double>* p = &sol;
  solution_data.add(p, "solution");
  
  dealii::AnyData data;
  dealii::Vector<double>* rhs = &res;
  data.add(rhs, "RHS");
  dealii::AnyData residual_data;
  residual(data, residual_data);
  dealii::deallog << "Residual " << res.l2_norm() << std::endl;
  solver(solution_data, data);
  app.error(errors, solution_data, error);
}

template <int dim>
void
iterative_solve_and_error(dealii::BlockVector<double>& errors,
		AmandusApplicationSparse<dim> &app,
		dealii::Algorithms::Operator<dealii::Vector<double> >& solver,
		const AmandusIntegrator<dim>& error)
{
  dealii::Vector<double> sol;
  
  app.setup_system();
  app.setup_vector(sol);
  solver.notify(dealii::Algorithms::Events::remesh);
  
  dealii::AnyData solution_data;
  dealii::Vector<double>* p = &sol;
  solution_data.add(p, "solution");
  
  dealii::AnyData data;
  solver(solution_data, data);
  app.error(errors, solution_data, error);
}

/**
 * @ingroup Verification
 */
template <int dim>
void
verify_residual(unsigned int n_refinements,
		AmandusApplicationSparse<dim> &app,
		AmandusIntegrator<dim>& matrix_integrator,
		AmandusIntegrator<dim>& residual_integrator)
{
  dealii::Vector<double> seed;
  dealii::Vector<double> diff;
  
  for (unsigned int s=0;s<n_refinements;++s)
    {
      app.refine_mesh(true);
  
      app.setup_system();
      app.setup_vector(seed);
      app.setup_vector(diff);
      
      for (unsigned int i=0;i<seed.size();++i)
	seed(i) = dealii::Utilities::generate_normal_random_number(0., 1.);
      
      dealii::AnyData diff_data;
      dealii::Vector<double>* p = &diff;
      diff_data.add(p, "diff");
      
      dealii::AnyData data;
      dealii::Vector<double>* rhs = &seed;
      data.add(rhs, "Newton iterate");
      
      app.assemble_matrix(data, matrix_integrator);
      app.verify_residual(diff_data, data, residual_integrator);
      app.output_results(s, &diff_data);
      
      dealii::deallog << "Difference " << diff.l2_norm() << std::endl;
    }
}

/**
 * @ingroup Verification
 */
template <int dim>
void
verify_theta_residual(unsigned int n_refinements,
		      AmandusApplicationSparse<dim> &app,
		      AmandusIntegrator<dim>& matrix_integrator,
		      AmandusIntegrator<dim>& residual_integrator)
{
  dealii::Vector<double> seed;
  dealii::Vector<double> prev;
  dealii::Vector<double> diff;
  
  for (unsigned int s=0;s<n_refinements;++s)
    {
      app.refine_mesh(true);
  
      app.setup_system();
      app.setup_vector(seed);
      app.setup_vector(prev);
      app.setup_vector(diff);
      
      for (unsigned int i=0;i<seed.size();++i)
	seed(i) = dealii::Utilities::generate_normal_random_number(0., 1.);
      for (unsigned int i=0;i<prev.size();++i)
	prev(i) = dealii::Utilities::generate_normal_random_number(0., 1.);
      
      double dt = .73;
      
      dealii::AnyData diff_data;
      diff_data.add(&diff, "diff");
      
      dealii::AnyData data;
      data.add(&dt, "Timestep");
      data.add(&seed, "Newton iterate");
      
      matrix_integrator.extract_data(data);
      residual_integrator.extract_data(data);
      
      app.assemble_matrix(data, matrix_integrator);
      app.verify_residual(diff_data, data, residual_integrator);
      app.output_results(s, &diff_data);
      
      dealii::deallog << "Difference " << diff.l2_norm() << std::endl;
    }
}

#endif





