// $Id$

#ifndef __readiff_matrix_h
#define __readiff_matrix_h

#include <deal.II/meshworker/integration_info.h>
#include <integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <brusselator/parameters.h>

using namespace dealii;
using namespace LocalIntegrators;


/**
 * Integrators for ReactionDiffusion problems
 *
 * These integrators deal with the equations
 *
 * \f{align*}{
 * 0 &= u' - \alpha\Delta u - B - u^2 v + (A+1) u \\
 * 0 &= v' - \alpha\Delta v - Au + u^2 v.
 * \f}
 *
 * A parameter set can be found in G. Adomian: The Diffusion
 * ReactionDiffusion Equation, Computers & Mathematics with Applications,
 * 29(5), pp. 1-3, 1995 with
 * \f{align*}{
 * A &= 3.4\\
 * B &= 1\\
 * \alpha &= .002.
 * \f}
 *
 */
namespace ReactionDiffusion
{
  /**
   * The derivative of the residual operator in ImplicitResidual
   * consists of 4 matrices, namely:
   *
   * \f{align*}{
   * \partial_u r_u(w) & = w - \theta\Delta t \bigl( \alpha\Delta w +
   * 2 uvw - (A+1)w \\
   * \partial_v r_u(w) & = - \theta\Delta t u^2 w \\
   * \partial_v r_v(w) & = w - \theta\Delta t \bigl( \alpha\Delta w -
   * u^2 \bigr) \\
   * \partial_u r_v(w) & = \theta\Delta t 2uvw
   * \f}
   */
  template <int dim>
  class Matrix : public AmandusIntegrator<dim>
  {
    public:
      Matrix(const Parameters& par);
    
      virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
			MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const;

    private:
      SmartPointer<const Parameters, class Matrix<dim> > parameters;
  };


  template <int dim>
  Matrix<dim>::Matrix(const Parameters& par)
		  :
		  parameters(&par)
  {
    this->use_boundary = false;
    this->use_face = true;
    this->input_vector_names.push_back("Newton iterate");
  }


  template <int dim>
  void Matrix<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
  {
    AssertDimension (dinfo.n_matrices(), 4);
    // Assert (info.values.size() >0, ExcInternalError());
    
    const Parameters& p = *parameters;
    Laplace::cell_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0),
    			 parameters->alpha1);
    Laplace::cell_matrix(dinfo.matrix(3,false).matrix, info.fe_values(0),
    			 parameters->alpha2);
    
    AssertDimension(info.values[0].size(), 2);
    AssertDimension(info.values[0][0].size(), info.fe_values(0).n_quadrature_points);
    AssertDimension(info.values[0][1].size(), info.fe_values(0).n_quadrature_points);
    std::vector<double> Du_ru(info.fe_values(0).n_quadrature_points);
    std::vector<double> Dv_ru(info.fe_values(0).n_quadrature_points);
    std::vector<double> Dv_rv(info.fe_values(0).n_quadrature_points);
    std::vector<double> Du_rv(info.fe_values(0).n_quadrature_points);
    for (unsigned int k=0;k<Du_ru.size();++k)
      {
	const double u = info.values[0][0][k];
	const double v = info.values[0][1][k];
	Du_ru[k] = p.B1 + p.D1*v + 2.*p.E1*u;
	Dv_ru[k] = p.C1 + p.D1*u + 2.*p.F1*v;
	Dv_rv[k] = p.C2 + p.D2*u + 2.*p.F2*v;
	Du_rv[k] = p.B2 + p.D2*v + 2.*p.E2*u;
      }
    L2::weighted_mass_matrix(dinfo.matrix(0,false).matrix, info.fe_values(0), Du_ru);
    L2::weighted_mass_matrix(dinfo.matrix(1,false).matrix, info.fe_values(0), Dv_ru);
    L2::weighted_mass_matrix(dinfo.matrix(2,false).matrix, info.fe_values(0), Dv_rv);
    L2::weighted_mass_matrix(dinfo.matrix(3,false).matrix, info.fe_values(0), Du_rv);
  }


  template <int dim>
  void Matrix<dim>::boundary(
    MeshWorker::DoFInfo<dim>&,
    typename MeshWorker::IntegrationInfo<dim>&) const
  {}


  template <int dim>
  void Matrix<dim>::face(
    MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
    MeshWorker::IntegrationInfo<dim>& info1, MeshWorker::IntegrationInfo<dim>& info2) const
  {
    const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
    const Parameters& p = *parameters;
    Laplace::ip_matrix(dinfo1.matrix(0,false).matrix, dinfo1.matrix(0,true).matrix, 
		       dinfo2.matrix(0,true).matrix, dinfo2.matrix(0,false).matrix,
		       info1.fe_values(0), info2.fe_values(0),
		       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
		       p.alpha1);
    Laplace::ip_matrix(dinfo1.matrix(3,false).matrix, dinfo1.matrix(3,true).matrix, 
		       dinfo2.matrix(3,true).matrix, dinfo2.matrix(3,false).matrix,
		       info1.fe_values(0), info2.fe_values(0),
		       Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
		       p.alpha2);
  }
}


#endif
