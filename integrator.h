/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __amandus_integrator_h
#define __amandus_integrator_h

#include <deal.II/meshworker/local_integrator.h>

template <int dim>
class AmandusIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
{
  public:
    AmandusIntegrator ();
    double timestep;
};

template <int dim>
AmandusIntegrator<dim>::AmandusIntegrator ()
		:
		timestep(0.)
{}

#endif

