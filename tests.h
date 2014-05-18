/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __tests_h
#define __tests_h

#include <deal.II/base/logstream.h>
#include <deal.II/base/named_data.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>

#include <amandus.h>

/**
 * @ingroup Verification
 */
template <int dim>
void
verify_residual(unsigned int n_refinements,
		AmandusApplication<dim> &app,
		const AmandusIntegrator<dim>& matrix_integrator,
		const AmandusIntegrator<dim>& residual_integrator)
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
      
      dealii::NamedData<dealii::Vector<double>* > diff_data;
      dealii::Vector<double>* p = &diff;
      diff_data.add(p, "diff");
      
      dealii::NamedData<dealii::Vector<double>* > data;
      dealii::Vector<double>* rhs = &seed;
      data.add(rhs, "Newton iterate");
      
      app.assemble_matrix(data, matrix_integrator);
      app.verify_residual(diff_data, data, residual_integrator);
      diff_data.add(rhs, "rhs");
      app.output_results(s, &diff_data);
      
      dealii::deallog << "Difference " << diff.l2_norm() << std::endl;
    }
}

#endif
