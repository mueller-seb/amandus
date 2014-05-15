/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __apps_h
#define __apps_h

#include <deal.II/base/logstream.h>
#include <deal.II/base/named_data.h>
#include <deal.II/lac/vector.h>

#include <amandus.h>

/**
 *
 *
 * @ingroup apps
 */
template <int dim>
void
global_refinement_linear_loop(unsigned int n_steps,
			      AmandusApplicationSparse<dim> &app,
			      dealii::Algorithms::Operator<dealii::Vector<double> >& solver,
			      dealii::Algorithms::Operator<dealii::Vector<double> >& residual,
			      const dealii::MeshWorker::LocalIntegrator<dim>* error = 0,
			      const dealii::MeshWorker::LocalIntegrator<dim>* estimator = 0)
{
  dealii::Vector<double> res;
  dealii::Vector<double> sol;
  
  for (unsigned int s=0;s<n_steps;++s)
    {
      dealii::deallog << "Step " << s << std::endl;
      app.refine_mesh(true);
      solver.notify(dealii::Algorithms::Events::remesh);
      app.setup_system();
      app.setup_vector(res);
      app.setup_vector(sol);
      
      dealii::NamedData<dealii::Vector<double>* > solution_data;
      dealii::Vector<double>* p = &sol;
      solution_data.add(p, "solution");
      
      dealii::NamedData<dealii::Vector<double>* > data;
      dealii::Vector<double>* rhs = &res;
      data.add(rhs, "RHS");
      dealii::NamedData<dealii::Vector<double>* > residual_data;
      residual(data, residual_data);
      dealii::deallog << "Residual " << res.l2_norm() << std::endl;
      solver(solution_data, data);
      if (error != 0)
	{
	  app.error(solution_data, *error, 5);
       	}

      if (estimator != 0)
	{
	  dealii::deallog << "Error::Estimate: "
			  << app.estimate(solution_data, *estimator)
			  << std::endl;
	}
      
      app.output_results(s, &solution_data);
    }

}


/**
 *
 *
 * @ingroup apps
 */
template <int dim>
void
global_refinement_nonlinear_loop(unsigned int n_steps,
			      AmandusApplicationSparse<dim> &app,
			      dealii::Algorithms::Operator<dealii::Vector<double> >& solve,
			      const dealii::MeshWorker::LocalIntegrator<dim>* error = 0,
			      const dealii::MeshWorker::LocalIntegrator<dim>* estimator = 0)
{
  dealii::Vector<double> res;
  dealii::Vector<double> sol;
  
  for (unsigned int s=0;s<n_steps;++s)
    {
      dealii::deallog << "Step " << s << std::endl;
      app.refine_mesh(true);
      solve.notify(dealii::Algorithms::Events::remesh);
      app.setup_system();
      app.setup_vector(sol);
      sol = 0.;
      
      dealii::NamedData<dealii::Vector<double>* > solution_data;
      dealii::Vector<double>* p = &sol;
      solution_data.add(p, "solution");
      
      dealii::NamedData<dealii::Vector<double>* > data;
      solve(solution_data, data);
      if (error != 0)
	{
	  app.error(solution_data, *error, 5);
       	}
      
      if (estimator != 0)
	{
	  dealii::deallog << "Error::Estimate: "
			  << app.estimate(solution_data, *estimator)
			  << std::endl;
	}
      
      app.output_results(s, &solution_data);
    }
}



#endif
