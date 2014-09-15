#ifndef __darcy_estimator_h
#define __darcy_estimator_h

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/algorithms/any_data.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <integrator.h>

namespace Darcy
{
  template <int dim>
    class Postprocessor : public AmandusIntegrator<dim>
  {
    public:
      Postprocessor(const dealii::DoFHandler<dim>& pp_dofh,
                    const dealii::DoFHandler<dim>& solution_dofh,
                    const dealii::TensorFunction<2, dim>& weight);
      void postprocess(dealii::Vector<double>& pp,
                       const dealii::Vector<double>& solution);

      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                            dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
                        dealii::MeshWorker::DoFInfo<dim>& dinfo2,
                        dealii::MeshWorker::IntegrationInfo<dim>& info1,
                        dealii::MeshWorker::IntegrationInfo<dim>& info2) const;
    private:
      void assemble_local_system(const dealii::FEValuesBase<dim>& fe_vals) const;
      void assemble_local_rhs(const dealii::FEValuesBase<dim>& fe_vals) const;
      void weighted_stiffness_matrix(dealii::FullMatrix<double>& result,
                                     const dealii::FEValuesBase<dim>& fe_vals,
                                     const dealii::TensorFunction<2, dim>&
                                     weight) const;
      void mixed_mass_matrix(dealii::FullMatrix<double>& result,
                             const dealii::FEValuesBase<dim>& fe_vals) const;

      const dealii::SmartPointer<const dealii::DoFHandler<dim> > pp_dofh;
      const dealii::SmartPointer<const dealii::TensorFunction<2, dim> > weight;

      const dealii::MappingQ1<dim> mapping;
      dealii::MeshWorker::DoFInfo<dim> dof_info;
      dealii::MeshWorker::IntegrationInfoBox<dim> info_box;

      const dealii::SmartPointer<const dealii::DoFHandler<dim> >
        approximation_dofh;
      dealii::SmartPointer<const dealii::Vector<double> > approximate_solution;
      dealii::SmartPointer<dealii::FEValues<dim> > approximation_fe_vals;
      const dealii::FEValuesExtractors::Vector velocity;
      const dealii::FEValuesExtractors::Scalar pressure;

      mutable dealii::FullMatrix<double> local_stiffness;
      mutable dealii::FullMatrix<double> local_mixed_mass;
      mutable dealii::FullMatrix<double> local_system;
      mutable dealii::FullMatrix<double> inverse_local_system;
      mutable dealii::Vector<double> local_rhs;
      mutable dealii::Vector<double> local_result;
      mutable std::vector<dealii::Tensor<2, dim> > local_weight_values;
      mutable std::vector<dealii::Tensor<1, dim > > approximation_velocity_values;
      mutable std::vector<double> approximation_pressure_values;
  };

  template <int dim>
    Postprocessor<dim>::Postprocessor(
        const dealii::DoFHandler<dim>& pp_dofh,
        const dealii::DoFHandler<dim>& solution_dofh,
        const dealii::TensorFunction<2, dim>& weight) 
    :
      pp_dofh(&pp_dofh),
      weight(&weight),
      dof_info(pp_dofh),
      approximation_dofh(&solution_dofh),
      velocity(0),
      pressure(dim)

    {
      info_box.initialize_update_flags();
      info_box.add_update_flags_all(dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_quadrature_points |
                                    dealii::update_JxW_values);
      info_box.initialize(pp_dofh.get_fe(),
                          mapping);

      this->use_cell = true;
      this->use_boundary = false;
      this->use_face = false;
    }

  template <int dim>
    void Postprocessor<dim>::postprocess(dealii::Vector<double>& pp,
                                         const dealii::Vector<double>& solution)
    {
      dealii::AnyData out;
      out.add<dealii::Vector<double>* >(&pp, "result");
      dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > 
        assembler;
      assembler.initialize(out);
      approximate_solution = &solution;
      dealii::FEValues<dim>* approximation_fe_vals_ptr = 
        new dealii::FEValues<dim>(
          approximation_dofh->get_fe(),
          info_box.cell_quadrature,
          dealii::update_values);
      approximation_fe_vals = approximation_fe_vals_ptr;
      dealii::MeshWorker::integration_loop(pp_dofh->begin_active(),
                                           pp_dofh->end(),
                                           dof_info,
                                           info_box,
                                           *this,
                                           assembler);
      approximation_fe_vals = 0;
      delete approximation_fe_vals_ptr;
      approximate_solution = 0;
    }

  template <int dim>
    void Postprocessor<dim>::cell(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {
      assemble_local_system(info.fe_values(0));
      assemble_local_rhs(info.fe_values(0));

      inverse_local_system.reinit(info.fe_values(0).dofs_per_cell + 1,
                                  info.fe_values(0).dofs_per_cell + 1);
      inverse_local_system.invert(local_system);
      local_result.reinit(info.fe_values(0).dofs_per_cell + 1);
      inverse_local_system.vmult(local_result,
                                 local_rhs);
      for(unsigned int i = 0; i < info.fe_values(0).dofs_per_cell; ++i)
      {
        dinfo.vector(0).block(0)(i) = local_result(i);
      }
    }

  template <int dim>
    void Postprocessor<dim>::boundary(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {}

  template <int dim>
    void Postprocessor<dim>::face(
        dealii::MeshWorker::DoFInfo<dim>& dinfo1,
        dealii::MeshWorker::DoFInfo<dim>& dinfo2,
        dealii::MeshWorker::IntegrationInfo<dim>& info1,
        dealii::MeshWorker::IntegrationInfo<dim>& info2) const
    {}

  template <int dim>
    void Postprocessor<dim>::assemble_local_system(
        const dealii::FEValuesBase<dim>& fe_vals) const
    {
      local_stiffness.reinit(fe_vals.dofs_per_cell, fe_vals.dofs_per_cell);
      weighted_stiffness_matrix(
          local_stiffness, fe_vals, *weight);
      
      local_mixed_mass.reinit(1, fe_vals.dofs_per_cell);
      mixed_mass_matrix(
          local_mixed_mass, fe_vals);

      local_system.reinit(fe_vals.dofs_per_cell + 1,
                          fe_vals.dofs_per_cell + 1);

      for(unsigned int i = 0; i < fe_vals.dofs_per_cell; ++i)
      {
        for(unsigned int j = 0; j < fe_vals.dofs_per_cell; ++j)
        {
          local_system.set(i, j, local_stiffness(i, j));
        }
      }
      for(unsigned int j = 0; j < fe_vals.dofs_per_cell; ++j)
      {
        local_system.set(fe_vals.dofs_per_cell, j,
                         local_mixed_mass(0, j));
        local_system.set(j, fe_vals.dofs_per_cell,
                         local_mixed_mass(0, j));
      }
    }

  template <int dim>
    void Postprocessor<dim>::weighted_stiffness_matrix(
        dealii::FullMatrix<double>& result,
        const dealii::FEValuesBase<dim>& fe_vals,
        const dealii::TensorFunction<2, dim>& weight) const
    {
      AssertDimension(fe_vals.dofs_per_cell, result.m());
      AssertDimension(fe_vals.dofs_per_cell, result.n());
      local_weight_values.resize(fe_vals.n_quadrature_points);
      weight.value_list(fe_vals.get_quadrature_points(),
                        local_weight_values);

      double dx;
      for(unsigned int q = 0; q < fe_vals.n_quadrature_points; ++q)
      {
        dx = fe_vals.JxW(q);
        for(unsigned int i = 0; i < fe_vals.dofs_per_cell; ++i)
        {
          for(unsigned int j = 0; j < fe_vals.dofs_per_cell; ++j)
          {
            for(unsigned int k = 0; k < dim; ++k)
            {
              for(unsigned int l = 0; l < dim; ++l)
              {
                result(i, j) += (local_weight_values[q][k][l] *
                                 fe_vals.shape_grad(j, q)[k] *
                                 fe_vals.shape_grad(i, q)[l] *
                                 dx);
              }
            }
          }
        }
      }
    }

  template <int dim>
    void Postprocessor<dim>::mixed_mass_matrix(
        dealii::FullMatrix<double>& result,
        const dealii::FEValuesBase<dim>& fe_vals) const
    {
      AssertDimension(1, result.m());
      AssertDimension(fe_vals.dofs_per_cell, result.n());

      double dx;
      for(unsigned int q = 0; q < fe_vals.n_quadrature_points; ++q)
      {
        dx = fe_vals.JxW(q);
        for(unsigned int j = 0; j < fe_vals.dofs_per_cell; ++j)
        {
          result(0, j) += (fe_vals.shape_value(j, q) *
                           1 *
                           dx);
        }
      }
    }

  template <int dim>
    void Postprocessor<dim>::assemble_local_rhs(
        const dealii::FEValuesBase<dim>& fe_vals) const
    {
      local_rhs.reinit(fe_vals.dofs_per_cell + 1);

      /*
      Assert(fe_vals.get_cell()->get_triangulation() ==
             approximation_dofh->get_tria(),
             dealii::ExcInternalError());
             */

      typename dealii::DoFHandler<dim>::active_cell_iterator cell(
          &(fe_vals.get_cell()->get_triangulation()),
          fe_vals.get_cell()->level(),
          fe_vals.get_cell()->index(),
          approximation_dofh);
      approximation_fe_vals->reinit(cell);
      approximation_velocity_values.resize(fe_vals.n_quadrature_points);
      (*approximation_fe_vals)[velocity].get_function_values(
          *approximate_solution,
          approximation_velocity_values);
      approximation_pressure_values.resize(fe_vals.n_quadrature_points);
      (*approximation_fe_vals)[pressure].get_function_values(
          *approximate_solution,
          approximation_pressure_values);

      double dx;
      for(unsigned int q = 0; q < fe_vals.n_quadrature_points; ++q)
      {
        dx = fe_vals.JxW(q);
        for(unsigned int i = 0; i < fe_vals.dofs_per_cell; ++i)
        {
          for(unsigned int k = 0; k < dim; ++k)
          {
            local_rhs(i) += (-1 *
                             approximation_velocity_values[q][k] *
                             fe_vals.shape_grad(i, q)[k] *
                             dx);
          }
        }
        local_rhs(fe_vals.dofs_per_cell) += (approximation_pressure_values[q] *
                                             1 *
                                             dx);
      }
    }


  /**
   * Provides the local integrator interface for Amandus to calculate the
   * a posteriori error estimate for Darcy's equation. The estimator uses a
   * postprocessing of the approximate solution.
   */
  /*
  template <int dim>
    class Estimator : public AmandusIntegrator<dim>
  {
    public:
      Estimator(const dealii::Function<dim>& exact_solution);
      Estimator(const dealii::Function<dim>& exact_solution,
                const dealii::TensorFunction<2, dim>& weight);
      ~Estimator();

      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                            dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
                        dealii::MeshWorker::DoFInfo<dim>& dinfo2,
                        dealii::MeshWorker::IntegrationInfo<dim>& info1,
                        dealii::MeshWorker::IntegrationInfo<dim>& info2) const;
    private:
      void init();

      const dealii::SmartPointer<const dealii::Function<dim> > exact_solution;

      const dealii::TensorFunction<2, dim>* const owned_weight;
      dealii::SmartPointer<const dealii::TensorFunction<2, dim> > weight;
  };

  template <int dim>
    Estimator<dim>::Estimator(
        const dealii::Function<dim>& exact_solution) :
      exact_solution(&exact_solution),
      owned_weight(new IdentityTensorFunction<dim>),
      weight(owned_weight)
  {
    init();
  }

  template <int dim>
    Estimator<dim>::Estimator(
        const dealii::Function<dim>& exact_solution,
        const dealii::TensorFunction<2, dim>& weight) :
      exact_solution(&exact_solution),
      owned_weight(0),
      weight(&weight)
  {
    init();
  }


  template <int dim>
    Estimator<dim>::~Estimator()
    {
      if(owned_weight != 0)
      {
        weight = 0;
        delete owned_weight;
      }
    }

  template <int dim>
    void Estimator<dim>::init()
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
    void Estimator<dim>::cell(
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

  template <int dim>
    void Estimator<dim>::boundary(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const 
    {
    }

  template <int dim>
    void Estimator<dim>::face(
        dealii::MeshWorker::DoFInfo<dim>& dinfo1,
        dealii::MeshWorker::DoFInfo<dim>& dinfo2,
        dealii::MeshWorker::IntegrationInfo<dim>& info1, 
        dealii::MeshWorker::IntegrationInfo<dim>& info2) const 
    {
    }
  */



    /*
  template <int dim>
    class PostprocessorIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
  {
    public:
      PostprocessorIntegrator(const dealii::TensorFunction<2, dim>& weight);
      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                            dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
                        dealii::MeshWorker::DoFInfo<dim>& dinfo2,
                        dealii::MeshWorker::IntegrationInfo<dim>& info1,
                        dealii::MeshWorker::IntegrationInfo<dim>& info2) const;
    private:
      dealii::UpdateFlags u_flags;
      dealii::SmartPointer<const dealii::TensorFunction<2, dim> > weight;
      dealii::SmartPointer<const dealii::Triangulation<dim> > triangulation;
      dealii::SmartPointer<const dealii::DoFHandler<dim> > 
        approximation_dof_handler;

      dealii::FullMatrix<double> local_stiffness;
      dealii::FullMatrix<double> local_mixed_mass;
      dealii::FullMatrix<double> local_system;
  };

  template <int dim>
    PostprocessorIntegrator<dim>::PostprocessorIntegrator(
        const dealii::TensorFunction<2, dim>& weight) :
      u_flags(
          dealii::update_values |
          dealii::update_gradients |
          dealii::update_JxW_values),
      weight(&weight)

    {
      this->use_cell = true;
      this->use_boundary = false;
      this->use_face = false;
    }

  template <int dim>
    PostprocessorIntegrator<dim>::cell(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const 
    {
      Assert(dinfo.vector(0).n_blocks() == 2, ExcInvalidState());
      Assert(dinfo.vector(0)(0).size() == info.fe_values(0).dofs_per_cell,
             ExcInvalidState());
      Assert(dinfo.vector(0)(1).size() == info.fe_values(1).dofs_per_cell,
             ExcInvalidState());

      const dealii::FEValuesBase<dim> pp_fe_values = &info.fe_values(0);
      const dealii::FeValuesBase<dim> multiplier_fe_values = &info.fe_values(1);
      dealii::BlockVector<double>& local_solution = dinfo.vector(0);

      const unsigned int total_dofs_per_cell = (
          pp_fe_values.dofs_per_cell + multiplier_fe_values.dofs_per_cell);

      // init data structures
      local_stiffness.reinit(pp_fe_values.dofs_per_cell);
      local_mixed_mass.reinit(multiplier_fe_values.dofs_per_cell,
                              pp_fe_values.dofs_per_cell);
      local_system.reinit(total_dofs_per_cell);
      local_rhs.reinit(total_dofs_per_cell);

      // assemble blocks
      weighted_stiffness_matrix(
          local_stiffness, pp_fe_values, *weight);
      mixed_mass_matrix(
          local_mixed_mass, multiplier_fe_values, pp_fe_values);

      // assemble system
      assemble_local_block_system(local_system, local_stiffness, local_mixed_mass);

      // assemble rhs
      Assert(dinfo.cell->get_triangulation == *triangulation, 
             dealii::ExcInvalidState());
      Assert(approximation_fe_values.n_quadrature_points == 
             pp_fe_values.n_quadrature_points, 
             ExcInvalidState());
      // get values of approximate solution in quadrature points on current
      // cell
      dealii::DoFHandler<dim>::active_cell_iterator cell(
          triangulation, 
          dinfo.cell->level(), 
          dinfo.cell->index(), 
          approximation_dof_handler);
      approximation_fe_values.reinit(cell);
      approximation_fe_values.get_function_values(
          approximate_solution,
          approximation_values);

      // assemble rhs
      for(unsigned int i = 0; i < pp_fe_values.dofs_per_cell; ++i)
      {
        rhs(i) = 0;
        for(unsigned int q = 0; q < pp_fe_values.n_quadrature_points; ++q)
        {
          for(unsigned int j = 0; j < dim; ++j)
          {
            rhs(i) += (-1 * 
                       approximation_values[q](j) * 
                       pp_fe_values.shape_grad(i, q)[j]);
          }
        }
      }
      for(unsigned int i = 0; i < multiplier_fe_values.dofs_per_cell; ++i)
      {
        rhs(i) = 0;
        for(unsigned int q = 0; q < pp_fe_values.n_quadrature_points; ++q)
        {
          rhs(i) += (approximation_values[q](dim) *
                     multiplier_fe_values.shape_value(i, q));
        }
      }

      // solve local system
      
    }
  
  template <int dim>
    void PostprocessorIntegrator<dim>::boundary(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const 
    {}

  template <int dim>
    void PostprocessorIntegrator<dim>::face(
        dealii::MeshWorker::DoFInfo<dim>& dinfo1,
        dealii::MeshWorker::DoFInfo<dim>& dinfo2,
        dealii::MeshWorker::IntegrationInfo<dim>& info1,
        dealii::MeshWorker::IntegrationInfo<dim>& info2) const
    {}

  */

  /**
   * assemble (a, b^t; b, 0) into system
   */
    /*
  template <int dim>
    void PostprocessorIntegrator<dim>::assemble_local_block_system(
        dealii::FullMatrix<double>& system,
        const dealii::FullMatrix<double>& a,
        const dealii::FullMatrix<double>& b)
    {
      Assert(a.m() == b.n(), ExcInvalidState());
      Assert(a.n() == b.m(), ExcInvalidState());
      // assemble the local system
      for(unsigned int i = 0; i < a.m(); ++i)
      {
        for(unsigned int j = 0; j < a.n(); ++j)
        {
          local_system.set(i, j, a(i, j));
        }
      }

      for(unsigned int i = 0; i < b.m(); ++i)
      {
        for(unsigned int j = 0; j < b.n(); ++j)
        {
          local_system.set(a.m() + i, j,
                           b(i, j));
          local_system.set(j, a.n() + i,
                           b(i, j));
        }
      }
    }
    */


}

#endif
