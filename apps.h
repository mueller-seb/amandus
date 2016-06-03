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
#include <deal.II/algorithms/any_data.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <amandus/amandus.h>
#include <amandus/adaptivity.h>
#include <iomanip>
#include <sstream>
#include <algorithm>

/**
 * @file
 * @brief Global refinement linear loop shows convergence table on console
 *
 * @ingroup apps
 */
template <int dim>
void
global_refinement_linear_loop(unsigned int n_steps,
			      AmandusApplicationSparse<dim> &app,
			      dealii::Algorithms::OperatorBase& solver,
			      dealii::Algorithms::OperatorBase& residual,
			      const AmandusIntegrator<dim>* error = 0,
			      AmandusIntegrator<dim>* estimator = 0,
			      const dealii::Function<dim>* initial_vector = 0,
			      const bool boundary_projection = false)
{
  dealii::Vector<double> res;
  dealii::Vector<double> sol;
  dealii::BlockVector<double> errors;
	if (error != 0)
		errors.reinit(error->n_errors());
  dealii::ConvergenceTable convergence_table;

  for (unsigned int s=0;s<n_steps;++s)
    {
      dealii::deallog << "Step " << s << std::endl;
      app.refine_mesh(true);
      solver.notify(dealii::Algorithms::Events::remesh);
      app.setup_system();
      app.setup_vector(res);
      app.setup_vector(sol);

      if (initial_vector != 0)
	{
	  if(boundary_projection)
	    {
	      app.update_vector_inhom_boundary(sol, *initial_vector, boundary_projection);	      
	    }
	  else
	    {
	      dealii::QGauss<dim> quadrature(app.dofs().get_fe().tensor_degree()+2);
	      dealii::VectorTools::project(app.dofs(),
					   app.hanging_nodes(), quadrature, *initial_vector, sol);
	      /* dealii::VectorTools::interpolate(app.dofs(), *initial_vector, sol); */
	    }
	}

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
          app.error(errors, solution_data, *error);
          for (unsigned int i=0;i<errors.n_blocks();++i)
	    {
              dealii::deallog << "Error(" << i << "): " << errors.block(i).l2_norm() << std::endl;
            }
          for(unsigned int i=0; i<errors.n_blocks();++i)
            {
              std::string  err_name {"Error("};
              err_name += std::to_string( i );
              err_name += ")";
              convergence_table.add_value(err_name, errors.block(i).l2_norm());
              convergence_table.evaluate_convergence_rates(err_name, dealii::ConvergenceTable::reduction_rate_log2);
              convergence_table.set_scientific(err_name, 1);
            }
       	}

      if (estimator != 0)
	{
	  dealii::deallog << "Error::Estimate: "
			  << app.estimate(solution_data, *estimator)
			  << std::endl;
	}
      dealii::AnyData out_data;
      out_data.merge(solution_data);
      if (error != 0)
        for (unsigned int i=0; i<errors.n_blocks(); ++i)
        {
          std::string  err_name {"Error("};
          err_name += std::to_string( i );
          err_name += ")";
          out_data.add(&errors.block(i),err_name);
        }
      app.output_results(s, &out_data);
      std::cout << std::endl;
      convergence_table.write_text(std::cout);
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
                                 dealii::Algorithms::OperatorBase &solve,
                                 const AmandusIntegrator<dim> *error = 0,
                                 AmandusIntegrator<dim> *estimator = 0,
                                 const dealii::Function<dim> *initial_vector = 0,
                                 const dealii::Function<dim> *inhom_boundary = 0,
				 const bool boundary_projection = false)
{
  dealii::Vector<double> res;
  dealii::Vector<double> sol;
  dealii::BlockVector<double> errors;
	if (error != 0)
		errors.reinit(error->n_errors());
  dealii::ConvergenceTable convergence_table;

  for (unsigned int s=0;s<n_steps;++s)
    {
      dealii::deallog << "Step " << s << std::endl;
      app.refine_mesh(true);
      solve.notify(dealii::Algorithms::Events::remesh);
      app.setup_system();
      app.setup_vector(sol);

      if (initial_vector != 0)
	{
	  dealii::QGauss<dim> quadrature(app.dofs().get_fe().tensor_degree()+2);
	  dealii::VectorTools::project(app.dofs(),
				       app.hanging_nodes(), quadrature, *initial_vector, sol);
	}

      if(inhom_boundary != 0)
        app.update_vector_inhom_boundary(sol, *inhom_boundary, boundary_projection);

      dealii::AnyData solution_data;
      solution_data.add(&sol, "solution");

      dealii::AnyData data;
      solve(solution_data, data);
      if (error != 0)
	{
          app.error(errors, solution_data, *error);
          for (unsigned int i=0;i<errors.n_blocks();++i)
	    {
              dealii::deallog << "Error(" << i << "): " << errors.block(i).l2_norm() << std::endl;
            }
          for(unsigned int i=0; i<errors.n_blocks();++i)
            {
              std::string  err_name {"Error("};
              err_name += std::to_string( i );
              err_name += ")";
              convergence_table.add_value(err_name, errors.block(i).l2_norm());
              convergence_table.evaluate_convergence_rates(err_name, dealii::ConvergenceTable::reduction_rate_log2);
              convergence_table.set_scientific(err_name, 1);
            }
       	}

      if (estimator != 0)
	{
	  dealii::deallog << "Error::Estimate: "
			  << app.estimate(solution_data, *estimator)
			  << std::endl;
	}

      dealii::AnyData out_data;
      out_data.merge(solution_data);
      if (error != 0)
        for (unsigned int i=0; i<errors.n_blocks(); ++i)
        {
          std::string  err_name {"Error("};
          err_name += std::to_string( i );
          err_name += ")";
          out_data.add(&errors.block(i),err_name);
        }
      app.output_results(s, &out_data);
      std::cout << std::endl;
      convergence_table.write_text(std::cout);
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
				  dealii::Algorithms::OperatorBase& solve)
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


/**
 * @file
 * @brief Adaptive refinement linear loop shows convergence table on console
 *
 * @ingroup apps_test
 */
template <int dim>
void
adaptive_refinement_linear_loop(unsigned int max_dofs,
                                AmandusApplicationSparse<dim> &app,
                                dealii::Triangulation<dim>& tria,
                                dealii::Algorithms::OperatorBase &solver,
                                dealii::Algorithms::OperatorBase &right_hand_side,
                                AmandusIntegrator<dim> &estimator,
                                AmandusRefineStrategy<dim> &mark,
                                const AmandusIntegrator<dim> *error = 0)
{
  dealii::Vector<double> rhs;
  dealii::Vector<double> sol;
  dealii::BlockVector<double> errors;
  if (error != 0)
    errors.reinit(error->n_errors());
  dealii::ConvergenceTable convergence_table;
  unsigned int step=0;
  
  // initial setup
  solver.notify(dealii::Algorithms::Events::initial);
  app.setup_system();
  app.setup_vector(sol);
  dealii::AnyData solution_data;
  solution_data.add(&sol, "solution");
  dealii::AnyData data;
  EstimateRemesher<dealii::Vector<double>, dim> remesh(app,tria,mark,estimator);
  
  while (true)
    {
      dealii::deallog << "Step " << step++ << std::endl;
      
      // solve
      app.setup_vector(rhs);
      dealii::AnyData rhs_data;
      rhs_data.add(&rhs, "RHS");
      right_hand_side(rhs_data, data);
      app.constraints().set_zero(sol);
      solver(solution_data, rhs_data);

      // error
      dealii::deallog << "Dofs: " << app.dofs().n_dofs() << std::endl;
      convergence_table.add_value("dofs",app.dofs().n_dofs());
      if (error != 0)
        {
          app.error(errors, solution_data, *error);
          for (unsigned int i=0; i<errors.n_blocks(); ++i)
            {
              dealii::deallog << "Error(" << i << "): " << errors.block(i).l2_norm() << std::endl;
            }
          for (unsigned int i=0; i<errors.n_blocks(); ++i)
            {
              std::string  err_name {"Error("};
              err_name += std::to_string( i );
              err_name += ")";
              convergence_table.add_value(err_name, errors.block(i).l2_norm());
              convergence_table.evaluate_convergence_rates(err_name, "dofs",dealii::ConvergenceTable::reduction_rate_log2,1);
              convergence_table.set_scientific(err_name, 1);
            }
        }

      // estimator
      const double est = app.estimate(solution_data, estimator);
      dealii::deallog << "Estimate: " << est << std::endl;
      convergence_table.add_value("estimate", est);
      convergence_table.evaluate_convergence_rates("estimate", "dofs",dealii::ConvergenceTable::reduction_rate_log2,1);

      // output
      dealii::AnyData out_data;
      out_data.merge(solution_data);
      if (error != 0)
        for (unsigned int i=0; i<errors.n_blocks(); ++i)
          {
            std::string  err_name {"Error("};
            err_name += std::to_string( i );
            err_name += ")";
            out_data.add(&errors.block(i),err_name);
          }
      dealii::Vector<double> indicators;
      indicators = app.indicators();
      out_data.add(&indicators,"estimator");
      app.output_results(step, &out_data);
      std::cout << std::endl;
      convergence_table.write_text(std::cout);

      // stop
      if (app.dofs().n_dofs()>=max_dofs) break;

      // mark & refine using remesher
      remesh(solution_data,data);
      solver.notify(dealii::Algorithms::Events::remesh);
    }
}

/**
 * @file
 * @brief Adaptive refinement non linear loop shows convergence table on console
 *
 * @ingroup apps_test
 */
template <int dim>
void adaptive_refinement_nonlinear_loop(
    unsigned int max_dofs, AmandusApplicationSparse<dim> &app,
    dealii::Triangulation<dim> &tria, dealii::Algorithms::OperatorBase &solver,
    AmandusIntegrator<dim> &estimator, AmandusRefineStrategy<dim> &mark,
    const AmandusIntegrator<dim> *error = 0,
    const dealii::Function<dim> *inhom_boundary = 0) {
  dealii::Vector<double> sol;
  dealii::BlockVector<double> errors;
  if (error != 0)
    errors.reinit(error->n_errors());
  dealii::ConvergenceTable convergence_table;
  unsigned int step = 0;

  // initial setup
  solver.notify(dealii::Algorithms::Events::initial);
  app.setup_system();
  app.setup_vector(sol);
  dealii::AnyData solution_data;
  solution_data.add(&sol, "solution");
  dealii::AnyData data;
  EstimateRemesher<dealii::Vector<double>, dim> remesh(app, tria, mark,
                                                       estimator);

  while (true) {
    dealii::deallog << "Step " << step++ << std::endl;

    // solve
    if (inhom_boundary != 0)
      app.update_vector_inhom_boundary(sol, *inhom_boundary);
    solver(solution_data, data);

    // error
    dealii::deallog << "Dofs: " << app.dofs().n_dofs() << std::endl;
    convergence_table.add_value("dofs", app.dofs().n_dofs());
    if (error != 0) {
      app.error(errors, solution_data, *error);
      for (unsigned int i = 0; i < errors.n_blocks(); ++i) {
        dealii::deallog << "Error(" << i << "): " << errors.block(i).l2_norm()
                        << std::endl;
      }
      for (unsigned int i = 0; i < errors.n_blocks(); ++i) {
        std::string err_name{"Error("};
        err_name += std::to_string(i);
        err_name += ")";
        convergence_table.add_value(err_name, errors.block(i).l2_norm());
        convergence_table.evaluate_convergence_rates(
            err_name, "dofs", dealii::ConvergenceTable::reduction_rate_log2, 1);
        convergence_table.set_scientific(err_name, 1);
      }
    }

    // estimator
    const double est = app.estimate(solution_data, estimator);
    dealii::deallog << "Estimate: " << est << std::endl;
    convergence_table.add_value("estimate", est);
    convergence_table.evaluate_convergence_rates(
        "estimate", "dofs", dealii::ConvergenceTable::reduction_rate_log2, 1);

    // output
    dealii::AnyData out_data;
    out_data.merge(solution_data);
    if (error != 0)
      for (unsigned int i = 0; i < errors.n_blocks(); ++i) {
        std::string err_name{"Error("};
        err_name += std::to_string(i);
        err_name += ")";
        out_data.add(&errors.block(i), err_name);
      }
    dealii::Vector<double> indicators;
    indicators = app.indicators();
    out_data.add(&indicators, "estimator");
    app.output_results(step, &out_data);
    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    // stop
    if (app.dofs().n_dofs() >= max_dofs)
      break;

    // mark & refine using remesher
    remesh(solution_data, data);
    solver.notify(dealii::Algorithms::Events::remesh);
  }
}

/**
 * @file
 * @brief Adaptive refinement eigenvalue loop for symmetric problems with real eigenvalues.
 *        Shows convergence table on console and refines for the n-th real eigenvalue
 *        (only supports the computation of one single simple eigenvalue).
 * @ingroup apps_test
 */

template <int dim>
void
adaptive_refinement_eigenvalue_loop(
  unsigned int max_dofs, unsigned int n_eigenvalue, AmandusApplicationSparse<dim>& app,
  dealii::Algorithms::OperatorBase& solver, AmandusIntegrator<dim>& estimator,
  AmandusRefineStrategy<dim>& mark,
  const double exact_eigenvalue = std::nan(NULL),
  const unsigned int k = 0)
{
  // compute n+k eigenvalues and select the n-th
  // eigenvalue after sorting
  const unsigned int n_vectors = n_eigenvalue + k;
  std::vector<std::complex<double>> eigenvalues(n_vectors);
  std::vector<dealii::Vector<double>> eigenvectors(2 * n_vectors);
  dealii::ConvergenceTable convergence_table;
  std::vector<std::pair<unsigned int, double> > sort_eigenvalues(n_vectors);
  solver.notify(dealii::Algorithms::Events::initial);
  dealii::AnyData solution_data;
  solution_data.add(&eigenvalues, "eigenvalues");
  solution_data.add(&eigenvectors, "eigenvectors");
  unsigned int step = 0;

  while (true)
  {
    dealii::deallog << "Step " << step++ << std::endl;

    // setup
    app.setup_system();
    for (unsigned int i = 0; i < eigenvectors.size(); ++i)
      app.setup_vector(eigenvectors[i]);

    // solve
    dealii::AnyData data;
    solver(solution_data, data);

    // sort eigenvalues
    for (unsigned int i = 0; i < n_vectors; ++i)
    {
      sort_eigenvalues[i].first = i;
      sort_eigenvalues[i].second = eigenvalues[i].real();
    }
    sort(sort_eigenvalues.begin(), sort_eigenvalues.end(), [](auto& left, auto& right) {
      return left.second < right.second;
    });
    const unsigned eigenvalue_idx = sort_eigenvalues[n_eigenvalue - 1].first;

    // error for n-th (real) eigenvalue
    dealii::deallog << "Dofs: " << app.dofs().n_dofs() << std::endl;
    convergence_table.add_value("dofs", app.dofs().n_dofs());
    if (std::isfinite(exact_eigenvalue))
    {
      const double error = std::abs(exact_eigenvalue - eigenvalues[eigenvalue_idx].real());
      dealii::deallog << "EV-Error: " << error << std::endl;
      convergence_table.add_value("EV-Error", error);
      convergence_table.evaluate_convergence_rates(
        "EV-Error", "dofs", dealii::ConvergenceTable::reduction_rate_log2, 1);
      convergence_table.set_scientific("EV-Error", 1);
    }

    // estimator for n-th (real) eigenvalue
    dealii::AnyData last_sol_data;
    dealii::Vector<double> solution = eigenvectors[eigenvalue_idx];
    last_sol_data.add(&solution, "solution");
    const double ev = eigenvalues[eigenvalue_idx].real();
    last_sol_data.add(&ev, "ev");
    const double est = std::pow(app.estimate(last_sol_data, estimator), 2);
    dealii::deallog << "Estimate: " << est << std::endl;
    convergence_table.add_value("estimate", est);
    convergence_table.evaluate_convergence_rates(
      "estimate", "dofs", dealii::ConvergenceTable::reduction_rate_log2, 1);

    // output
    dealii::AnyData out_data;
    out_data.add(&eigenvectors[eigenvalue_idx],
                 std::string("ev") + std::to_string(eigenvalue_idx));
    dealii::Vector<double> indicators;
    indicators = app.indicators();
    out_data.add(&indicators, "estimator");
    for (unsigned int i = 0; i < n_vectors; ++i)
      dealii::deallog << "Eigenvalue " << '\t' << std::setprecision(15)
                      << sort_eigenvalues[i].second << std::endl;
    convergence_table.add_value("ev", eigenvalues[eigenvalue_idx].real());
    app.output_results(step, &out_data);
    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    // stop
    if (app.dofs().n_dofs() >= max_dofs)
      break;
    // mark
    mark(app.indicators());
    // refine
    app.refine_mesh(false);
    solver.notify(dealii::Algorithms::Events::remesh);
  }
}
#endif
