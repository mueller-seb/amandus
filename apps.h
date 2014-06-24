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
			      const AmandusIntegrator<dim>* error = 0,
			      const AmandusIntegrator<dim>* estimator = 0,
			      const dealii::Function<dim>* initial_vector = 0)
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

      if (initial_vector)
	dealii::VectorTools::interpolate(app.dofs(), *initial_vector, sol);
      
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
			      const AmandusIntegrator<dim>* error = 0,
			      const AmandusIntegrator<dim>* estimator = 0,
			      const dealii::Function<dim>* initial_vector = 0)
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
      if (initial_vector)
	{
	  dealii::QGauss<dim> quadrature(app.dofs().get_fe().tensor_degree()+1);
	  dealii::VectorTools::project(app.dofs(), app.hanging_nodes(), quadrature, *initial_vector, sol);
	}
      else
	sol = 0.;
      
      dealii::AnyData solution_data;
      solution_data.add(&sol, "solution");
      
      dealii::AnyData data;
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
