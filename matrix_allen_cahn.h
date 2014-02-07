// $Id: matrix_allen_cahn.h 1397 2014-02-02 22:42:29Z kanschat $

#ifndef __matrix_allen_cahn_h
#define __matrix_allen_cahn_h

#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;


/**
 * Integrator for AllenCahn problems
 */
template <int dim>
class AllenCahnMatrix : public MeshWorker::LocalIntegrator<dim>
{
  public:
    AllenCahnMatrix(double diffusion);
    
    virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
    virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
    virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
		      MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const;

  private:
    double D;
};


template <int dim>
AllenCahnMatrix<dim>::AllenCahnMatrix(double diffusion)
		:
		D(diffusion)
{
  this->input_vector_names.push_back("Newton iterate");
}


  template <int dim>
void AllenCahnMatrix<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  AssertDimension (dinfo.n_matrices(), 1);
//  Assert (info.values.size() >0, ExcInternalError());
  Laplace::cell_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), D);

  if (info.values.size() > 0)
    {
      AssertDimension(info.values[0][0].size(), info.fe_values(0).n_quadrature_points);
      std::vector<double> fu(info.fe_values(0).n_quadrature_points);
      for (unsigned int k=0;k<fu.size();++k)
	{
	  const double u = info.values[0][0][k];
	  fu[k] = 3.*u*u-1.;
	}
      L2::weighted_mass_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), fu);
    }
}


template <int dim>
void AllenCahnMatrix<dim>::boundary(
  MeshWorker::DoFInfo<dim>& dinfo,
  typename MeshWorker::IntegrationInfo<dim>& info) const
{
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0),
  			  Laplace::compute_penalty(dinfo, dinfo, deg, deg), D);
}


template <int dim>
void AllenCahnMatrix<dim>::face(
  MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
  MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Laplace::ip_matrix(dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix, 
  		     dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
  		     info1.fe_values(0), info2.fe_values(0),
  		     Laplace::compute_penalty(dinfo1, dinfo2, deg, deg), D);
}


#endif
