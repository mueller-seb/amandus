/**
 * @file
 * <ul>
 * <li> Stationary Biot equations</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Newton solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */

#include <amandus/apps.h>
#include <amandus/biot/matrix.h>
#include <biot/parameters.h>
#include <amandus/biot/residual.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;

template <int dim>
class Startup : public dealii::Function<dim>
{
public:
  Startup();
  virtual void vector_value_list(const std::vector<Point<dim>>& points,
                                 std::vector<Vector<double>>& values) const override;
  virtual void vector_values(const std::vector<Point<dim>>& points,
                             std::vector<std::vector<double>>& values) const override;
};

template <int dim>
Startup<dim>::Startup()
  : Function<dim>(2 * dim + 1)
{
}

template <int dim>
void
Startup<dim>::vector_value_list(const std::vector<Point<dim>>& points,
                                std::vector<Vector<double>>& values) const
{
  AssertDimension(points.size(), values.size());

  for (unsigned int k = 0; k < points.size(); ++k)
  {
    const Point<dim>& p = points[k];
    if (p(0) < 0.)
    {
      values[k](0) = -.0;
      values[k](dim) = 1. - p(1) * p(1) - 0. / 3.;
    }
    else if (p(0) > 0.)
    {
      values[k](0) = .0;
      //	  values[k](dim) = 1.;
    }
  }
}

template <int dim>
void
Startup<dim>::vector_values(const std::vector<Point<dim>>& points,
                            std::vector<std::vector<double>>& values) const
{
  AssertVectorVectorDimension(values, this->n_components, points.size());

  for (unsigned int k = 0; k < points.size(); ++k)
  {
    const Point<dim>& p = points[k];
    if (p(0) < 0.)
    {
      values[0][k] = -.0;
      values[dim][k] = 1. - p(1) * p(1) - 0. / 3.;
    }
    else if (p(0) > 0.)
    {
      values[0][k] = .0;
      //	  values[k](dim) = 1.;
    }
  }
}

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);

  AmandusParameters param;
  ::Biot::Parameters::declare_parameters(param);
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  std::unique_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1, true);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  Startup<d> startup;

  ::Biot::Parameters parameters;
  parameters.parse_parameters(param);
  ::Biot::TestMatrix<d> matrix_integrator(parameters);
  ::Biot::TestResidual<d> rhs_integrator(parameters, startup);
  rhs_integrator.input_vector_names.push_back("Newton iterate");

  AmandusUMFPACK<d> app(tr, *fe);
  app.parse_parameters(param);
  ComponentMask inflow(2 * d + 1, true);
  ComponentMask free(2 * d + 1, false);
  ComponentMask outflow(2 * d + 1, false);
  inflow.set(2 * d, false);
  for (unsigned int dd = 0; dd < d; ++dd)
  {
    // Set no flow on upper and lower boundary
    free.set(d + dd, true);
    // Prescribe displacement at outflow
    outflow.set(dd, true);
  }

  app.set_boundary(0, inflow);
  app.set_boundary(1, outflow);
  app.set_boundary(2, free);
  app.set_boundary(3, free);

  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);

  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dofs());

  Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.parse_parameters(param);

  newton.initialize(newout);
  //  newton.debug_vectors = true;

  global_refinement_nonlinear_loop<d>(7, app, newton, 0, 0, &startup);
}
