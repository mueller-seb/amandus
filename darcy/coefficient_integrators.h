#ifndef __coefficient_integrators_h
#define __coefficient_integrators_h

#include <integrator.h>
#include "coefficient_parameters.h"

#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>

namespace dealii
{
  namespace LocalIntegrators
  {
    namespace L2
    {
      template <int dim>
        void tensor_weighted_mass_matrix(
            FullMatrix<double>& M,
            const FEValuesBase<dim>& fe,
            const TensorFunction<2, dim>& weight)
        {
          const unsigned int n_dofs = fe.dofs_per_cell;
          const unsigned int n_components = fe.get_fe().n_components();
          const unsigned int n_quadrature_points = fe.n_quadrature_points;
          std::vector<Tensor<2, dim> > weight_values(n_quadrature_points);
          weight.value_list(fe.get_quadrature_points(), weight_values);

          for(unsigned int q = 0; q < n_quadrature_points; ++q)
          {
            const double dx = fe.JxW(q);

            for(unsigned int i = 0; i < n_dofs; ++i)
            {
              for(unsigned int j = 0; j < n_dofs; ++j)
              {
                for(unsigned int k1 = 0; k1 < n_components; ++k1)
                {
                  M(i, j) += (dx *
                              weight_values[q][0][0] * 
                              fe.shape_value_component(i, q, k1) *
                              fe.shape_value_component(j, q, k1));
                  /*
                  for(unsigned int k2 = 0; k2 < n_components; ++k2)
                  {
                    M(i, j) += (dx *
                                weight_values[q][k1][k2] * 
                                fe.shape_value_component(i, q, k2) *
                                fe.shape_value_component(j, q, k1));
                  }
                  */
                }
              }
            }
          }
        }
    }
  }
}
            

namespace DarcyCoefficient
{
  template <int dim>
    class SystemIntegrator : public AmandusIntegrator<dim>
  {
    public:
      SystemIntegrator(const Parameters<dim>& parameters);
      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                            dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
                        dealii::MeshWorker::DoFInfo<dim>& dinfo2,
                        dealii::MeshWorker::IntegrationInfo<dim>& info1,
                        dealii::MeshWorker::IntegrationInfo<dim>& info2) const;
    private:
      const SmartPointer<const TensorFunction<2, dim> > weight;
  };

  template <int dim>
    SystemIntegrator<dim>::SystemIntegrator(const Parameters<dim>& parameters) :
      weight(&parameters.inverse_coefficient_tensor)
    {
      this->use_boundary = false;
      this->use_face = false;
    }

  template <int dim>
    void SystemIntegrator<dim>::cell(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {
      AssertDimension(dinfo.n_matrices(), 4);

      dealii::LocalIntegrators::L2::tensor_weighted_mass_matrix(
          dinfo.matrix(0).matrix, info.fe_values(0),
          *weight);
      dealii::LocalIntegrators::Divergence::cell_matrix(
          dinfo.matrix(2).matrix, info.fe_values(0), info.fe_values(1));
      dinfo.matrix(1).matrix.copy_transposed(dinfo.matrix(2).matrix);
      dinfo.matrix(1).matrix *= -1.0;
    }

  template <int dim>
    void SystemIntegrator<dim>::boundary(
        MeshWorker::DoFInfo<dim>&,
        typename MeshWorker::IntegrationInfo<dim>&) const
    {}

  template <int dim>
    void SystemIntegrator<dim>::face(
        MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>&,
        MeshWorker::IntegrationInfo<dim>& info1, 
        MeshWorker::IntegrationInfo<dim>&)
    const
    {}


  template <int dim>
    class RHSIntegrator : public AmandusIntegrator<dim>
  {
    public:
      RHSIntegrator(const Parameters<dim>& parameters);
      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                            dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
                        dealii::MeshWorker::DoFInfo<dim>& dinfo2,
                        dealii::MeshWorker::IntegrationInfo<dim>& info1,
                        dealii::MeshWorker::IntegrationInfo<dim>& info2) const;
    private:
      SmartPointer<const MixedSolution<dim> > mixed_solution;
  };

  template <int dim>
    RHSIntegrator<dim>::RHSIntegrator(const Parameters<dim>& parameters) :
      mixed_solution(&parameters.mixed_solution)
  {
    this->use_cell = false;
    this->use_boundary = true;
    this->use_face = false;
  }

  template <int dim>
    void RHSIntegrator<dim>::cell(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {}

  template <int dim>
    void RHSIntegrator<dim>::boundary(
        MeshWorker::DoFInfo<dim>& dinfo,
        typename MeshWorker::IntegrationInfo<dim>& info) const
    {
      const unsigned int n_quadrature_points =
        info.fe_values(0).n_quadrature_points;

      std::vector<double> boundary_values(n_quadrature_points);
      mixed_solution->value_list(
          info.fe_values(0).get_quadrature_points(),
          boundary_values,
          dim);

      dealii::LocalIntegrators::Divergence::u_times_n_residual(
          dinfo.vector(0).block(0),
          info.fe_values(0),
          boundary_values,
          -1.0);
    }

  template <int dim>
    void RHSIntegrator<dim>::face(
        MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>&,
        MeshWorker::IntegrationInfo<dim>& info1, 
        MeshWorker::IntegrationInfo<dim>&)
    const
    {}
}

#endif
