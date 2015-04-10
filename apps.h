/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __apps_h
#define __apps_h

#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>

#include <amandus.h>
#include <iomanip>

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


/**
 *
 *
 * @ingroup apps
 */
template <int dim>
void
global_refinement_eigenvalue_loop(unsigned int n_steps,
				  unsigned int n_values,
				  AmandusApplicationSparse<dim> &app,
				  dealii::Algorithms::Operator<dealii::Vector<double> >& solve)
{
  std::vector<std::complex<double> > eigenvalues(n_values);
  std::vector<dealii::Vector<double> > eigenvectors(2*n_values);
  
  for (unsigned int s=0;s<n_steps;++s)
    {
      dealii::deallog << "Step " << s << std::endl;
      app.refine_mesh(true);
      solve.notify(dealii::Algorithms::Events::remesh);
      app.setup_system();
      for (unsigned int i=0;i<eigenvectors.size();++i)
	app.setup_vector(eigenvectors[i]);
      
      dealii::AnyData solution_data;
      solution_data.add(&eigenvalues, "eigenvalues");
      solution_data.add(&eigenvectors, "eigenvectors");
      
      dealii::AnyData data;
      solve(solution_data, data);

      dealii::AnyData out_data;
      for (unsigned int i=0;i<n_values;++i)
	{
	  out_data.add(&eigenvectors[i], std::string("ev") + std::to_string(i)
			    + std::string("re"));
	  out_data.add(&eigenvectors[n_values+i], std::string("ev") + std::to_string(i)
			    + std::string("im"));
	  dealii::deallog << "Eigenvalue " << i << '\t' << std::setprecision(15) << eigenvalues[i] << std::endl;
	}
      
      app.output_results(s, &out_data);
    }
}



#endif










