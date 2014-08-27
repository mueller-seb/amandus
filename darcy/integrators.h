#ifndef __darcy_integrators_h
#define __darcy_integrators_h

#include <integrator.h>

#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/divergence.h>

/**
 * Namespace containing integrators for mixed discretizations of Darcy's
 * equation.
 *
 * @ingroup integrators
 */
namespace Darcy
{
  /**
   * The mass matrix for a weighted L2 inner product of vector valued finite
   * elements, i.e.
   * \f[
   * M_{ij} = \int K \phi_j \phi_i
   * \f]
   * where \f$K\f$ is the weight and \f$(\phi_i)\f$ are the local shape
   * functions.
   */
  template <int dim>
    void weighted_mass_matrix(
        dealii::FullMatrix<double>& M,
        const dealii::FEValuesBase<dim>& fe,
        const dealii::TensorFunction<2, dim>& weight)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_components = fe.get_fe().n_components();
      AssertDimension(n_components, dim);
      const unsigned int n_quadrature_points = fe.n_quadrature_points;
      std::vector<dealii::Tensor<2, dim> > weight_values(n_quadrature_points);
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
              for(unsigned int k2 = 0; k2 < n_components; ++k2)
              {
                M(i, j) += (dx *
                            weight_values[q][k1][k2] * 
                            fe.shape_value_component(i, q, k2) *
                            fe.shape_value_component(j, q, k1));
              }
            }
          }
        }
      }
    }
            

  /**
   * Local system integrator for mixed discretization of Darcy's equation.
   * Assembling the local contribution to the system
   * \f[
   * \left( 
   * \begin{array}{cc}
   * (K^{-1} \psi_j, \psi_i)_{ij} & -(\nabla \cdot \psi_i, \phi_k)_{ik} \\
   * (\nabla \cdot \psi_j, \phi_k)_{kj} & 0 \\
   * \end{array}
   * \right)
   * \f]
   * The weight \f$K^{-1}\f$ is given as an argument to the constructor.
   * Notice that it is the _inverse_ of the diffusion tensor.
   */
  template <int dim>
    class SystemIntegrator : public AmandusIntegrator<dim>
  {
    public:
      SystemIntegrator(const dealii::TensorFunction<2, dim>& weight);
      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                            dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
                        dealii::MeshWorker::DoFInfo<dim>& dinfo2,
                        dealii::MeshWorker::IntegrationInfo<dim>& info1,
                        dealii::MeshWorker::IntegrationInfo<dim>& info2) const;
    private:
      const dealii::SmartPointer<const dealii::TensorFunction<2, dim> > weight;
  };

  template <int dim>
    SystemIntegrator<dim>::SystemIntegrator(
        const dealii::TensorFunction<2, dim>& weight) :
      weight(&weight)
    {
      this->use_cell = true;
      this->use_boundary = false;
      this->use_face = false;
    }

  template <int dim>
    void SystemIntegrator<dim>::cell(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {
      AssertDimension(dinfo.n_matrices(), 4);

      weighted_mass_matrix(
          dinfo.matrix(0).matrix, info.fe_values(0),
          *weight);
      dealii::LocalIntegrators::Divergence::cell_matrix(
          dinfo.matrix(2).matrix, info.fe_values(0), info.fe_values(1));
      dinfo.matrix(1).matrix.copy_transposed(dinfo.matrix(2).matrix);
      dinfo.matrix(1).matrix *= -1.0;
    }

  template <int dim>
    void SystemIntegrator<dim>::boundary(
        dealii::MeshWorker::DoFInfo<dim>&,
        dealii::MeshWorker::IntegrationInfo<dim>&) const
    {}

  template <int dim>
    void SystemIntegrator<dim>::face(
        dealii::MeshWorker::DoFInfo<dim>& dinfo1, dealii::MeshWorker::DoFInfo<dim>&,
        dealii::MeshWorker::IntegrationInfo<dim>& info1, 
        dealii::MeshWorker::IntegrationInfo<dim>&)
    const
    {}

  /**
   * Local integrator for a right hand side of a mixed discretization of
   * Darcy's equation corresponding to Dirichlet boundary conditions and a
   * divergence free flux, i.e.
   * \f[
   * \left( 
   * \begin{array}{c}
   * (\sigma, \psi_i)_{i} \\
   * (0)_{k} 
   * \end{array}
   * \right)
   * \f]
   * The boundary values \f$\sigma\f$ are given as an argument to the
   * constructor.
   */
  template <int dim>
    class RHSIntegrator : public AmandusIntegrator<dim>
  {
    public:
      RHSIntegrator(const dealii::Function<dim>& boundary_function);
      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                            dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
                        dealii::MeshWorker::DoFInfo<dim>& dinfo2,
                        dealii::MeshWorker::IntegrationInfo<dim>& info1,
                        dealii::MeshWorker::IntegrationInfo<dim>& info2) const;
    private:
      dealii::SmartPointer<const dealii::Function<dim> > boundary_function;
  };

  template <int dim>
    RHSIntegrator<dim>::RHSIntegrator(
        const dealii::Function<dim>& boundary_function) :
      boundary_function(&boundary_function)
  {
    AssertDimension(boundary_function.n_components, 1);
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
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {
      const unsigned int n_quadrature_points =
        info.fe_values(0).n_quadrature_points;

      std::vector<double> boundary_values(n_quadrature_points);
      boundary_function->value_list(
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
        dealii::MeshWorker::DoFInfo<dim>& dinfo1, dealii::MeshWorker::DoFInfo<dim>&,
        dealii::MeshWorker::IntegrationInfo<dim>& info1, 
        dealii::MeshWorker::IntegrationInfo<dim>&)
    const
    {}
}

#endif
