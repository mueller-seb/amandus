#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <vector>
#include <string>
#include <sstream>
#include <fstream>

namespace debug
{
  template <int dim>
    void output_solution(const dealii::Function<dim>& function,
                         const dealii::DoFHandler<dim>& dof_handler,
                         const dealii::Quadrature<dim>& quadrature,
                         dealii::ParameterHandler& param)
    {
      dealii::Vector<double> projection(dof_handler.n_dofs());

      dealii::ConstraintMatrix constraints;
      dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                      constraints);
      constraints.close();

      dealii::VectorTools::project(dof_handler,
                                   constraints,
                                   quadrature,
                                   function,
                                   projection);

      std::vector<std::string> names;
      std::string basename = "component";
      for(unsigned int component = 0; component < function.n_components; ++component)
      {
        std::ostringstream name;
        name << basename << component;
        names.push_back(name.str());
      }

      dealii::DataOut<dim> data_out;
      data_out.parse_parameters(param);

      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(projection,
                               names);
      data_out.build_patches();

      std::ostringstream filename;
      filename << "projected_solution" << data_out.default_suffix();
      std::ofstream output(filename.str().c_str());
      data_out.write(output);
    }
}
