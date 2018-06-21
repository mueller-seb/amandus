/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace debug
{
template <int dim>
void
output_solution(const dealii::Function<dim>& function, const dealii::DoFHandler<dim>& dof_handler,
                const dealii::Quadrature<dim>& quadrature, dealii::ParameterHandler& param)
{
  dealii::Vector<double> projection(dof_handler.n_dofs());

  dealii::AffineConstraints<double> constraints;
  dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  dealii::VectorTools::project(dof_handler, constraints, quadrature, function, projection);

  std::vector<std::string> names;
  std::string basename = "component";
  for (unsigned int component = 0; component < function.n_components; ++component)
  {
    std::ostringstream name;
    name << basename << component;
    names.push_back(name.str());
  }

  dealii::DataOut<dim> data_out;
  data_out.parse_parameters(param);

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(projection, names);
  data_out.build_patches(dof_handler.get_fe().tensor_degree());

  std::ostringstream filename;
  filename << "projected_solution" << data_out.default_suffix();
  std::ofstream output(filename.str().c_str());
  data_out.write(output);
}

template <int dim>
void
output_fe_data(const dealii::DoFHandler<dim>& dofh, const dealii::Vector<double> dof_vector,
               const std::string& name, dealii::ParameterHandler& param)
{
  std::vector<std::string> names;
  for (unsigned int c = 0; c < dealii::DoFTools::n_components(dofh); ++c)
  {
    std::ostringstream c_name;
    c_name << name << "component" << c;
    names.push_back(c_name.str());
  }

  dealii::DataOut<dim> data_out;
  data_out.parse_parameters(param);

  data_out.attach_dof_handler(dofh);
  data_out.add_data_vector(dof_vector, names);
  data_out.build_patches(dofh.get_fe().tensor_degree());

  std::ostringstream filename;
  filename << name << data_out.default_suffix();
  std::ofstream output(filename.str().c_str());
  data_out.write(output);
}

template <int dim>
void
output_errors(dealii::BlockVector<double> errors, const dealii::Triangulation<dim>& tria,
              unsigned int s, std::string prefix)
{
  dealii::DataOut<dim> data_out;
  data_out.attach_triangulation(tria);

  for (std::size_t i = 0; i < errors.n_blocks(); ++i)
  {
    std::ostringstream name;
    name << prefix << "_error" << i;
    data_out.add_data_vector(errors.block(i), name.str(), dealii::DataOut<dim>::type_cell_data);
  }
  data_out.build_patches();

  std::ostringstream filename;
  filename << prefix << "_errors-" << s << ".vtk";
  std::ofstream out_file(filename.str().c_str());
  data_out.write_vtk(out_file);
}
}
