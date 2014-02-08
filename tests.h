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
#include <deal.II/lac/vector.h>

#include "amandus.h"

template <int dim>
void
verify_residual(unsigned int n_refinements,
		AmandusApplication<dim> &app,
		const dealii::MeshWorker::LocalIntegrator<dim>& integrator)
{
  dealii::Vector<double> seed;
  dealii::Vector<double> diff;
  
  for (unsigned int s=0;s<n_refinements;++s)
    {
      app.refine_mesh(true);
    }

  app.notify(dealii::Algorithms::Events::remesh);
  app.setup_system();
  app.setup_vector(seed);
  app.setup_vector(diff);
      
  dealii::NamedData<dealii::Vector<double>* > diff_data;
  dealii::Vector<double>* p = &diff;
  diff_data.add(p, "diff");
      
  dealii::NamedData<dealii::Vector<double>* > data;
  dealii::Vector<double>* rhs = &seed;
  data.add(rhs, "Newton iterate");
  app.verify_residual(integrator, diff_data, data);
  app.output_results(n_refinements, &diff_data);
}

#endif
