/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __advectiondiffusion_matrix_h
#define __advectiondiffusion_matrix_h

#include <deal.II/meshworker/integration_info.h>
#include <integrator.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/advection.h>

using namespace dealii::MeshWorker;


namespace Advection
{
/**
 * Integrator for Advection-Diffusion problems.
 *
 * The distinction between stationary and instationary problems is
 * made by the variable AmandusIntegrator::timestep, which is
 * inherited from the base class. If this variable is zero, we solve a
 * stationary problem. If it is nonzero, we assemble for an implicit
 * scheme.
 *
 * @ingroup integrators
 */
  template <int dim>
  class Matrix : public AmandusIntegrator<dim>
  {
    public:
      Matrix(const Parameters& par, double faktor, std::vector<std::vector<double> > direction);
      
      virtual void cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
      virtual void boundary(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const;
      virtual void face(DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2,
			IntegrationInfo<dim>& info1, IntegrationInfo<dim>& info2) const;
    private:
      dealii::SmartPointer<const Parameters, class Matrix<dim> > parameters;
      double faktor;
      std::vector<std::vector<double> > direction;
  };


  template <int dim>
  Matrix<dim>::Matrix(const Parameters& par, double faktor, std::vector<std::vector<double> > direction)
    :
    parameters(&par),
    faktor(faktor),
    direction(direction) 
  {
    //this->input_vector_names.push_back("Newton iterate");
  }
  
  template <int dim>
  void Matrix<dim>::cell(DoFInfo<dim>& dinfo, IntegrationInfo<dim>& info) const
  {
    FullMatrix<double> &M1 = dinfo.matrix(0,false).matrix;
    FullMatrix<double> &M2 = dinfo.matrix(0,false).matrix;

    AssertDimension (dinfo.n_matrices(), 1);
    dealii::LocalIntegrators::Advection::
      cell_matrix(M1,
		  info.fe_values(0), info.fe_values(0), direction);
    
    Laplace::cell_matrix(M2, info.fe_values(0), faktor);

    FullMatrix<double> &M = dinfo.matrix(0,false).matrix;
    M.equ(1,M1,1,M2);  
  }
  
  
  template <int dim>
  void Matrix<dim>::boundary(
    DoFInfo<dim>& dinfo,
    IntegrationInfo<dim>& info) const
  {

    FullMatrix<double> &M1 = dinfo.matrix(0,false).matrix;
    FullMatrix<double> &M2 = dinfo.matrix(0,false).matrix;

    dealii::LocalIntegrators::Advection::
      upwind_value_matrix(M1,
			  info.fe_values(0), info.fe_values(0),
			  direction);
   
    const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
    Laplace::nitsche_matrix(M2, info.fe_values(0),
  			  Laplace::compute_penalty(dinfo, dinfo, deg, deg), faktor);



     FullMatrix<double> &M = dinfo.matrix(0,false).matrix;
    M.equ(1,M1,1,M2);
   

  }
  
  
  template <int dim>
  void Matrix<dim>::face(
    DoFInfo<dim>& dinfo1, DoFInfo<dim>& dinfo2,
    IntegrationInfo<dim>& info1, IntegrationInfo<dim>& info2) const
  {

    FullMatrix<double> &M11 = dinfo1.matrix(0,false).matrix;
    FullMatrix<double> &M12 = dinfo1.matrix(0,true).matrix;
    FullMatrix<double> &M13 = dinfo2.matrix(0,true).matrix;
    FullMatrix<double> &M14 = dinfo2.matrix(0,false).matrix;
    FullMatrix<double> &M21 = dinfo1.matrix(0,false).matrix;
    FullMatrix<double> &M22 = dinfo1.matrix(0,true).matrix;
    FullMatrix<double> &M23 = dinfo2.matrix(0,true).matrix;
    FullMatrix<double> &M24 = dinfo2.matrix(0,false).matrix;

    dealii::LocalIntegrators::Advection::upwind_value_matrix(
      M11,
      M12, 
      M13,
      M14,
      info1.fe_values(0), info2.fe_values(0),
      info1.fe_values(0), info2.fe_values(0),
      direction);

    const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
    Laplace::ip_matrix(	M21, 
		      	M22, 
  		     	M23, 
		     	M24,
  		     	info1.fe_values(0), 
			info2.fe_values(0),
  		     	Laplace::compute_penalty(dinfo1, dinfo2, deg, deg), faktor);


    FullMatrix<double> &M1 = dinfo1.matrix(0,false).matrix;
    FullMatrix<double> &M2 = dinfo1.matrix(0,true).matrix;
    FullMatrix<double> &M3 = dinfo2.matrix(0,true).matrix;
    FullMatrix<double> &M4 = dinfo2.matrix(0,false).matrix;
    M1.equ(1,M11,1,M21);
    M2.equ(1,M12,1,M22);
    M3.equ(1,M13,1,M23);
    M4.equ(1,M14,1,M24);


  }
}


#endif










