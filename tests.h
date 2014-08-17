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





