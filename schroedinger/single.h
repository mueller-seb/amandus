/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef schroedinger_single_h
#define schroedinger_single_h

#include <deal.II/meshworker/integration_info.h>
#include <integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

using namespace dealii;
using namespace LocalIntegrators;


/**
 * Integrators for Schroedinger problems.
 *
 * @ingroup integrators
 */
namespace SchroedingerIntegrators
{
  template <int dim>
  class Eigen : public AmandusIntegrator<dim>
  {
    public:
      virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
			MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const;
  };
  
  
  template <int dim>
  void Eigen<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
  {
    const FEValuesBase<dim>& fe = info.fe_values(0);
    FullMatrix<double>& M = dinfo.matrix(0,false).matrix;
    Laplace::cell_matrix(M, fe);
    const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_components = fe.get_fe().n_components();

      for (unsigned int k=0; k<fe.n_quadrature_points; ++k)
        {
          const double dx = fe.JxW(k);
          for (unsigned int i=0; i<n_dofs; ++i)
            {
              double Mii = 0.0;
              for (unsigned int d=0; d<n_components; ++d)
                Mii -= dx * 1./fe.quadrature_point(k).norm()
                       * fe.shape_value_component(i,k,d)
                       * fe.shape_value_component(i,k,d);

              M(i,i) += Mii;

              for (unsigned int j=i+1; j<n_dofs; ++j)
                {
                  double Mij = 0.0;
                  for (unsigned int d=0; d<n_components; ++d)
                    Mij -= dx * 1./fe.quadrature_point(k).norm()
                           * fe.shape_value_component(j,k,d)
                           * fe.shape_value_component(i,k,d);

                  M(i,j) += Mij;
                  M(j,i) += Mij;
                }
            }
        }

    
    
    if(dinfo.n_matrices()==2)
      L2::mass_matrix(dinfo.matrix(1,false).matrix, info.fe_values(0));
  }
  
  
  template <int dim>
  void Eigen<dim>::boundary(
    MeshWorker::DoFInfo<dim>& dinfo,
    typename MeshWorker::IntegrationInfo<dim>& info) const
  {
    const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
    Laplace::nitsche_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0),
			    Laplace::compute_penalty(dinfo, dinfo, deg, deg));
  }
  
  
  template <int dim>
  void Eigen<dim>::face(
    MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
    MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const
  {
    const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
    Laplace::ip_matrix(dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix, 
		       dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
		       info1.fe_values(0), info2.fe_values(0),
		       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));
  }
}

#endif
