// $Id$

#ifndef __curl_curl_h
#define __curl_curl_h

#include "matrix_curl_curl.h"
#include "amandus.h"

template <int dim>
class CurlCurl : public AmandusApplication<dim>
{
  public:
    typedef MeshWorker::IntegrationInfo<dim> CellInfo;
    
    CurlCurl(Triangulation<dim>& triangulation,
	     const FiniteElement<dim>& fe,
	     const MeshWorker::LocalIntegrator<dim>& matrix_integrator,
	     const MeshWorker::LocalIntegrator<dim>& rhs_integrator);
};


template <int dim>
CurlCurl<dim>::CurlCurl(
  Triangulation<dim>& triangulation,
  const FiniteElement<dim>& fe,
  const MeshWorker::LocalIntegrator<dim>& matrix_integrator,
  const MeshWorker::LocalIntegrator<dim>& rhs_integrator)
  : AmandusApplication<dim>(triangulation, fe, matrix_integrator, rhs_integrator)
{}

#endif
