/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __darcy_estimator_h
#define __darcy_estimator_h

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/algorithms/any_data.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

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
      void init_vector(dealii::Vector<double>& vector);
      void postprocess(dealii::Vector<double>& pp,
                       const dealii::Vector<double>& solution);

      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;

    private:
      void assemble_local_system(dealii::FullMatrix<double>& local_system,
                                 const dealii::FEValuesBase<dim>& fe_vals) const;
      void assemble_local_rhs(dealii::Vector<double>& local_rhs,
                              const dealii::FEValuesBase<dim>& fe_vals,
                              const dealii::Function<dim>&
                              approximation_function) const;
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
      approximation_dofh(&solution_dofh)
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
    void Postprocessor<dim>::init_vector(dealii::Vector<double>& vector)
    {
      vector.reinit(pp_dofh->n_dofs());
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
      dealii::MeshWorker::integration_loop(pp_dofh->begin_active(),
                                           pp_dofh->end(),
                                           dof_info,
                                           info_box,
                                           *this,
                                           assembler);
      approximate_solution = 0;
    }

  template <int dim>
    void Postprocessor<dim>::cell(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {
      dealii::FullMatrix<double>
        local_system(info.fe_values(0).dofs_per_cell + 1,
                     info.fe_values(0).dofs_per_cell + 1);
      assemble_local_system(local_system, info.fe_values(0));

      dealii::Functions::FEFieldFunction<dim>
        approximation_function(*approximation_dofh, *approximate_solution);
      typename dealii::DoFHandler<dim>::active_cell_iterator cell(
          &(info.fe_values(0).get_cell()->get_triangulation()),
          info.fe_values(0).get_cell()->level(),
          info.fe_values(0).get_cell()->index(),
          approximation_dofh);
      approximation_function.set_active_cell(cell);
      dealii::Vector<double> local_rhs(info.fe_values(0).dofs_per_cell + 1);
      assemble_local_rhs(local_rhs, info.fe_values(0), approximation_function);

      dealii::FullMatrix<double> inverse_local_system(
          info.fe_values(0).dofs_per_cell + 1,
          info.fe_values(0).dofs_per_cell + 1);
      inverse_local_system.invert(local_system);

      dealii::Vector<double> local_result(info.fe_values(0).dofs_per_cell + 1);
      inverse_local_system.vmult(local_result,
                                 local_rhs);

      for(unsigned int i = 0; i < info.fe_values(0).dofs_per_cell; ++i)
      {
        dinfo.vector(0).block(0)(i) = local_result(i);
      }
    }

  template <int dim>
    void Postprocessor<dim>::assemble_local_system(
        dealii::FullMatrix<double>& local_system,
        const dealii::FEValuesBase<dim>& fe_vals) const
    {
      dealii::FullMatrix<double> local_stiffness(fe_vals.dofs_per_cell,
                                                 fe_vals.dofs_per_cell);
      weighted_stiffness_matrix(
          local_stiffness, fe_vals, *weight);
      
      dealii::FullMatrix<double> local_mixed_mass(1, fe_vals.dofs_per_cell);
      mixed_mass_matrix(
          local_mixed_mass, fe_vals);

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
      std::vector<dealii::Tensor<2, dim> >
        local_weight_values(fe_vals.n_quadrature_points);
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
        dealii::Vector<double>& local_rhs,
        const dealii::FEValuesBase<dim>& fe_vals,
        const dealii::Function<dim>& approximation_function) const
    {
      std::vector<dealii::Vector<double> > approximation_values(
          fe_vals.n_quadrature_points, dealii::Vector<double>(dim + 1));
      approximation_function.vector_value_list(fe_vals.get_quadrature_points(),
                                               approximation_values);

      double dx;
      for(unsigned int q = 0; q < fe_vals.n_quadrature_points; ++q)
      {
        dx = fe_vals.JxW(q);
        for(unsigned int i = 0; i < fe_vals.dofs_per_cell; ++i)
        {
          for(unsigned int k = 0; k < dim; ++k)
          {
            local_rhs(i) += (-1 *
                             approximation_values[q][k] *
                             fe_vals.shape_grad(i, q)[k] *
                             dx);
          }
        }
        local_rhs(fe_vals.dofs_per_cell) += (approximation_values[q][dim] *
                                             1 *
                                             dx);
      }
    }


  template <int dim>
    class Interpolator : public dealii::MeshWorker::LocalIntegrator<dim>
  {
    public:
      Interpolator(const dealii::DoFHandler<dim>& input_dofh,
                   unsigned int interpolation_degree,
                   const dealii::TensorFunction<2, dim>& weight,
                   const dealii::Function<dim>* bdry = 0);

      const dealii::FE_Q<dim>& get_fe() const;
      const dealii::DoFHandler<dim>& get_dofh() const;
      void init_vector(dealii::Vector<double>& result);
      void interpolate(dealii::Vector<double>& result,
                       const dealii::Vector<double>& input);

      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;

    private:
      dealii::FE_Q<dim> fe;
      dealii::DoFHandler<dim> dofh;
      dealii::SmartPointer<const dealii::TensorFunction<2, dim> > weight;
      dealii::SmartPointer<const dealii::Function<dim> > bdry;
      dealii::SmartPointer<const dealii::DoFHandler<dim> > input_dofh;

      std::vector<double> normalization_map;

      dealii::SmartPointer<const dealii::Vector<double> > input_fe_vector;

      void init_normalization();
      double squared_max_ev(
          typename dealii::Triangulation<dim>::cell_iterator cell) const;
  };

  template <int dim>
    Interpolator<dim>::Interpolator(
        const dealii::DoFHandler<dim>& input_dofh,
        unsigned int interpolation_degree,
        const dealii::TensorFunction<2, dim>& weight,
        const dealii::Function<dim>* bdry) :
      fe(interpolation_degree),
      weight(&weight),
      bdry(bdry),
      input_dofh(&input_dofh)
    {
      dofh.initialize(input_dofh.get_tria(), fe);

      this->use_cell = true;
      this->use_boundary = false;
      this->use_face = false;
    }

  template <int dim>
    const dealii::FE_Q<dim>& Interpolator<dim>::get_fe() const
    {
      return fe;
    }

  template <int dim>
    const dealii::DoFHandler<dim>& Interpolator<dim>::get_dofh() const
    {
      return dofh;
    }

  template <int dim>
    void Interpolator<dim>::init_vector(dealii::Vector<double>& result)
    {
      result.reinit(dofh.n_dofs());
    }

  template <int dim>
    void Interpolator<dim>::interpolate(
        dealii::Vector<double>& result,
        const dealii::Vector<double>& input)
    {
      dofh.initialize(dofh.get_tria(), dofh.get_fe());
      result.reinit(dofh.n_dofs());
      init_normalization();

      dealii::AnyData out;
      out.add<dealii::Vector<double>* >(&result, "result");

      dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> >
        assembler;
      assembler.initialize(out);

      input_fe_vector = &input;

      dealii::Quadrature<dim> continuous_support_points(
          fe.get_unit_support_points());

      dealii::MappingQ1<dim> mapping;
      dealii::MeshWorker::DoFInfo<dim> dof_info(dofh);
      dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
      info_box.initialize_update_flags();
      info_box.add_update_flags_all(dealii::update_values |
                                    dealii::update_quadrature_points);
      info_box.initialize(fe, mapping);

      dealii::MeshWorker::integration_loop(
          dofh.begin_active(),
          dofh.end(),
          dof_info,
          info_box,
          *this,
          assembler);

      input_fe_vector = 0;

      if(bdry != 0)
      {
        std::map<dealii::types::global_dof_index, double> bdry_values;
        dealii::VectorTools::interpolate_boundary_values(
            dofh,
            0,
            *bdry,
            bdry_values);

        typedef typename std::map<dealii::types::global_dof_index,
                double>::const_iterator map_iterator;
        map_iterator bdry_pair = bdry_values.begin(),
                     bdry_end = bdry_values.end();
        for(; bdry_pair != bdry_end; ++bdry_pair) {
          result(bdry_pair->first) = bdry_pair->second;
        }
      }

    }

  template <int dim>
    void Interpolator<dim>::init_normalization()
    {
      normalization_map.assign(dofh.n_dofs(), 0.0);

      std::vector<unsigned int> global_dof_indices(fe.dofs_per_cell);

      for(typename dealii::DoFHandler<dim>::active_cell_iterator cell =
          dofh.begin_active();
          cell != dofh.end();
          ++cell)
      {
        cell->get_dof_indices(global_dof_indices);
        for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        {
          normalization_map[global_dof_indices[i]] += squared_max_ev(cell);
        }
      }
    }

  template <int dim>
    double Interpolator<dim>::squared_max_ev(
        typename dealii::Triangulation<dim>::cell_iterator cell) const
    {
      // TODO: only works for piecewise constant weight tensor
      dealii::Tensor<2, dim> cell_value = weight->value(cell->center());
      dealii::LAPACKFullMatrix<double> lapack_cell_value(dim, dim);
      for(unsigned int i = 0; i < dim; ++i)
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          lapack_cell_value(i, j) = cell_value[i][j];
        }
      }
      lapack_cell_value.compute_eigenvalues();
      double ev2 = std::abs(lapack_cell_value.eigenvalue(1));
      if(dim == 3)
      {
        ev2 = std::max(ev2, std::abs(lapack_cell_value.eigenvalue(2)));
      }
        
      return std::sqrt(std::max(std::abs(lapack_cell_value.eigenvalue(0)), ev2));
    }

  template <int dim>
    void Interpolator<dim>::cell(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {
      dealii::Quadrature<dim> continuous_support_points(
          dofh.get_fe().get_unit_support_points());

      dealii::FEValues<dim> input_fe_vals(
          input_dofh->get_fe(),
          continuous_support_points,
          dealii::update_values);
      typename dealii::DoFHandler<dim>::active_cell_iterator cell(
          &(dinfo.cell->get_triangulation()),
          dinfo.cell->level(),
          dinfo.cell->index(),
          input_dofh);
      input_fe_vals.reinit(cell);

      std::vector<double> input_vals(info.fe_values(0).dofs_per_cell);
      input_fe_vals.get_function_values(*input_fe_vector, input_vals);

      typename dealii::DoFHandler<dim>::active_cell_iterator cell_pp(
          &(dinfo.cell->get_triangulation()),
          dinfo.cell->level(),
          dinfo.cell->index(),
          &dofh);
      std::vector<unsigned int> global_dof_indices(fe.dofs_per_cell);
      cell_pp->get_dof_indices(global_dof_indices);

      double ev = squared_max_ev(dinfo.cell);

      for(unsigned int i = 0; i < info.fe_values(0).dofs_per_cell; ++i)
      {
        dinfo.vector(0).block(0)(i) = (
            ev * input_vals[i] /
            normalization_map[global_dof_indices[i]]);
      }
    }

  /**
   * Provides the local integrator interface for Amandus to calculate the
   * a posteriori error estimate for Darcy's equation. The estimator uses a
   * postprocessing of the approximate solution.
   */
  template <int dim>
    class Estimator : public AmandusIntegrator<dim>
  {
    public:
      class Parameters;

      Estimator(Parameters& parameters);

      void reinit(const dealii::Vector<double>& input);

      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;

    private:
      void init();

      dealii::SmartPointer<const Parameters> parameters;
      Postprocessor<dim> postprocessor;
      Interpolator<dim> interpolator;
      dealii::Vector<double> pp_vector;
      dealii::Vector<double> dc_pp_vector;
  };

  template <int dim>
    class Estimator<dim>::Parameters : public dealii::Subscriptor
    {
      public:
        Parameters(
            dealii::DoFHandler<dim>& pp_dofh,
            const dealii::DoFHandler<dim>& input_dofh,
            const dealii::TensorFunction<2, dim>& weight,
            const dealii::TensorFunction<2, dim>& i_weight,
            unsigned int interpolation_degree,
            const dealii::Function<dim>* bdry = 0);
        dealii::DoFHandler<dim>& pp_dofh;
        const dealii::DoFHandler<dim>& input_dofh;
        const dealii::TensorFunction<2, dim>& weight;
        const dealii::TensorFunction<2, dim>& i_weight;
        unsigned int interpolation_degree;
        const dealii::Function<dim>* bdry;
    };

  template <int dim>
    Estimator<dim>::Parameters::Parameters(
        dealii::DoFHandler<dim>& pp_dofh,
        const dealii::DoFHandler<dim>& input_dofh,
        const dealii::TensorFunction<2, dim>& weight,
        const dealii::TensorFunction<2, dim>& i_weight,
        unsigned int interpolation_degree,
        const dealii::Function<dim>* bdry) :
      pp_dofh(pp_dofh), input_dofh(input_dofh), weight(weight),
      i_weight(i_weight), interpolation_degree(interpolation_degree),
      bdry(bdry)
  {}


  template <int dim>
    Estimator<dim>::Estimator(
        Parameters& parameters) :
      parameters(&parameters),
      postprocessor(parameters.pp_dofh,
                    parameters.input_dofh,
                    parameters.weight),
      interpolator(parameters.pp_dofh,
                   parameters.interpolation_degree,
                   parameters.weight,
                   parameters.bdry)
  {
    init();
  }

  template <int dim>
    void Estimator<dim>::init()
    {
      this->use_cell = true;
      this->use_boundary = false;
      this->use_face = false;

      this->add_flags(dealii::update_JxW_values |
                      dealii::update_values |
                      dealii::update_quadrature_points);
    }

  template <int dim>
    void Estimator<dim>::reinit(const dealii::Vector<double>& input)
    {
      parameters->pp_dofh.initialize(
          parameters->pp_dofh.get_tria(), 
          parameters->pp_dofh.get_fe());
      postprocessor.init_vector(dc_pp_vector);
      postprocessor.postprocess(dc_pp_vector, input);
      interpolator.init_vector(pp_vector);
      interpolator.interpolate(pp_vector, dc_pp_vector);
    }

  template <int dim>
    void Estimator<dim>::cell(
        dealii::MeshWorker::DoFInfo<dim>& dinfo,
        dealii::MeshWorker::IntegrationInfo<dim>& info) const 
    {
      const dealii::FEValuesBase<dim>& velocity_fev = info.fe_values(0);
      const unsigned int n_quadrature_points = velocity_fev.n_quadrature_points;

      std::vector<dealii::Tensor<2, dim> > weight_values(n_quadrature_points);
      std::vector<dealii::Tensor<2, dim> > i_weight_values(n_quadrature_points);
      std::vector<dealii::Tensor<1, dim> > pp_gradients(n_quadrature_points);
      std::vector<std::vector<double> >& approximation_values = info.values[0];

      parameters->weight.value_list(velocity_fev.get_quadrature_points(),
                                     weight_values);
      parameters->i_weight.value_list(velocity_fev.get_quadrature_points(),
                                       i_weight_values);

      dealii::Functions::FEFieldFunction<dim> pp(interpolator.get_dofh(),
                                                 pp_vector);
      typename dealii::DoFHandler<dim>::active_cell_iterator cell(
          &(info.fe_values(0).get_cell()->get_triangulation()),
          info.fe_values(0).get_cell()->level(),
          info.fe_values(0).get_cell()->index(),
          &(interpolator.get_dofh()));
      pp.set_active_cell(cell);
      pp.gradient_list(velocity_fev.get_quadrature_points(), pp_gradients);

      double dx;
      for(unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        dx = velocity_fev.JxW(q);
        for(unsigned int i = 0; i < dim; ++i)
        {
          dinfo.value(0) += (
              2 * approximation_values[i][q] * pp_gradients[q][i]) * dx;
          for(unsigned int k = 0; k < dim; ++k)
          {
            dinfo.value(0) += (
                pp_gradients[q][i] * weight_values[q][i][k] * pp_gradients[q][k] +
                i_weight_values[q][i][k] * approximation_values[k][q] *
                approximation_values[i][q]) * dx;
          }
        }
      }
      dinfo.value(0) = std::sqrt(dinfo.value(0));
    }
      
}

#endif
