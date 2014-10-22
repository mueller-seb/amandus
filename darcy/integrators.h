/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __darcy_integrators_h
#define __darcy_integrators_h

#include <integrator.h>

#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
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
   * where \f$K\f$ is a tensor valued weight and \f$(\phi_i)\f$ are the
   * local shape functions.
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
   * Simple function returning the identity tensor. Used as default value
   * for the weight in the system matrix.
   */
  template <int dim>
    class IdentityTensorFunction : public dealii::TensorFunction<2, dim>
  {
    public:
      typedef typename dealii::TensorFunction<2, dim>::value_type value_type;
      IdentityTensorFunction();

      virtual value_type value(const dealii::Point<dim>& p) const;

      dealii::Tensor<2, dim> identity;
  };

  template <int dim>
    IdentityTensorFunction<dim>::IdentityTensorFunction()
    {
      for(unsigned int i = 0; i < dim; ++i)
      {
        identity[i][i] = 1.0;
      }
    }

  template <int dim>
    typename IdentityTensorFunction<dim>::value_type
    IdentityTensorFunction<dim>::value(const dealii::Point<dim>& p) const
    {
      return identity;
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
   * No argument implies the identity tensor as weight.
   */
  template <int dim>
    class SystemIntegrator : public AmandusIntegrator<dim>
  {
    public:
      SystemIntegrator();
      SystemIntegrator(const dealii::TensorFunction<2, dim>& weight);
      ~SystemIntegrator();

      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;
    private:
      void init();
      const dealii::TensorFunction<2, dim>* const weight_ptr;
      dealii::SmartPointer<const dealii::TensorFunction<2, dim> > weight;
  };

  template <int dim>
    SystemIntegrator<dim>::SystemIntegrator() : 
      weight_ptr(new IdentityTensorFunction<dim>),
      weight(weight_ptr)
  {
    init();
  }

  template <int dim>
    SystemIntegrator<dim>::SystemIntegrator(
        const dealii::TensorFunction<2, dim>& weight) :
      weight_ptr(0),
      weight(&weight)
  {
    init();
  }

  template <int dim>
    SystemIntegrator<dim>::~SystemIntegrator()
    {
      if(weight_ptr != 0) 
      {
        weight = 0;
        delete weight_ptr;
      }
    }

  template <int dim>
    void SystemIntegrator<dim>::init()
    {
      this->use_cell = true;
      this->use_boundary = false;
      this->use_face = false;

      this->add_flags(dealii::update_JxW_values |
                      dealii::update_values |
                      dealii::update_gradients |
                      dealii::update_quadrature_points);
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

      virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                            dealii::MeshWorker::IntegrationInfo<dim>& info) const;
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

    this->add_flags(dealii::update_JxW_values |
                    dealii::update_values |
                    dealii::update_quadrature_points);
  }

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


  /**
   * Error integrator computing the \f$L^2\f$ error w.r.t. the exact
   * solution given as the constructor's argument. Uses a weighted norm for
   * the velocity component if a weight is passed to the constructor. 
   */
  template <int dim>
    class ErrorIntegrator : public AmandusIntegrator<dim>
  {
    public:
      ErrorIntegrator(const dealii::Function<dim>& exact_solution);
      ErrorIntegrator(const dealii::Function<dim>& exact_solution,
                      const dealii::TensorFunction<2, dim>& weight);
      ~ErrorIntegrator();

      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;
    private:
      void init();

      const dealii::SmartPointer<const dealii::Function<dim> > exact_solution;

      const dealii::TensorFunction<2, dim>* const owned_weight;
      dealii::SmartPointer<const dealii::TensorFunction<2, dim> > weight;
  };

  template <int dim>
    ErrorIntegrator<dim>::ErrorIntegrator(
        const dealii::Function<dim>& exact_solution) :
      exact_solution(&exact_solution),
      owned_weight(new IdentityTensorFunction<dim>),
      weight(owned_weight)
  {
    init();
  }

  template <int dim>
    ErrorIntegrator<dim>::ErrorIntegrator(
        const dealii::Function<dim>& exact_solution,
        const dealii::TensorFunction<2, dim>& weight) :
      exact_solution(&exact_solution),
      owned_weight(0),
      weight(&weight)
  {
    init();
  }


  template <int dim>
    ErrorIntegrator<dim>::~ErrorIntegrator()
    {
      if(owned_weight != 0)
      {
        weight = 0;
        delete owned_weight;
      }
    }

  template <int dim>
    void ErrorIntegrator<dim>::init()
    {
      AssertDimension(exact_solution->n_components, dim + 1);
      this->use_cell = true;
      this->use_boundary = false;
      this->use_face = false;

      this->add_flags(dealii::update_JxW_values |
                      dealii::update_values |
                      dealii::update_gradients |
                      dealii::update_quadrature_points);

    }

  template <int dim>
    void ErrorIntegrator<dim>::cell(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const 
    {
      const dealii::FEValuesBase<dim>& velocity_fe_values = info.fe_values(0);
      const dealii::FEValuesBase<dim>& pressure_fe_values = info.fe_values(1);

      const std::vector<std::vector<double> >& 
        velocity_approximation = info.values[0];
      const std::vector<double>&
        pressure_approximation = info.values[0][dim];

      std::vector<std::vector<double> > velocity_exact(
          dim, std::vector<double>(velocity_fe_values.n_quadrature_points));
      std::vector<double> pressure_exact(pressure_fe_values.n_quadrature_points);
      for(unsigned int i = 0; i < dim; ++i)
      {
        exact_solution->value_list(velocity_fe_values.get_quadrature_points(),
                                   velocity_exact[i],
                                   i);
      }
      exact_solution->value_list(pressure_fe_values.get_quadrature_points(),
                                 pressure_exact,
                                 dim);

      std::vector<dealii::Tensor<2, dim> > weight_values(
          velocity_fe_values.n_quadrature_points);
      weight->value_list(velocity_fe_values.get_quadrature_points(),
                         weight_values);

      // L2 error of velocity
      double velocity_l2_error = 0;
      for(unsigned int i = 0; i < dim; ++i)
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          for(unsigned int q = 0; q < velocity_fe_values.n_quadrature_points; ++q)
          {
            velocity_l2_error += (
                weight_values[q][i][j] *
                (velocity_exact[j][q] - velocity_approximation[j][q]) *
                (velocity_exact[i][q] - velocity_approximation[i][q]) *
                velocity_fe_values.JxW(q));
          }
        }
      }
      dinfo.value(0) = std::sqrt(velocity_l2_error);

      // L2 error of pressure
      double pressure_l2_error = 0;
      for(unsigned int q = 0; q < pressure_fe_values.n_quadrature_points; ++q)
      {
        pressure_l2_error += (
            std::pow(pressure_exact[q] - pressure_approximation[q], 2) *
            pressure_fe_values.JxW(q));
      }
      dinfo.value(1) = std::sqrt(pressure_l2_error);
    }

}

#endif
