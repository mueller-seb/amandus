/**
 * @file
 *
 * @brief Stationay Elasicity model with homogeneous Dirichlet boundary conditions.
 * @ingroup Elasticitygroup
 *
 * The code has the follows the features:
 * <ul>
 * <li> Stationary Elasticity equations</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Newton solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 *
 * <h3> Boundary conditions </h3>
 *
 * We impose the displacement on the left and on the
 * right edge of the square domain.
 * In particular:
 *
 * \f{align*}
 * u_x &=-1 \qquad\qquad & \text{on}\ x=0,
 * \\
 * u_x &= 1 \qquad\qquad & \text{on}\ x=1,
 * \f}
 *
 */

#include <deal.II/fe/fe_tools.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <deal.II/base/function.h>
#include <amandus/apps.h>
#include <elasticity/parameters.h>
#include <amandus/elasticity/residual.h>
#include <amandus/elasticity/matrix.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;

template <int dim>
class Startup : public dealii::Function<dim>
{
public:
  Startup();
  virtual void vector_value_list(const std::vector<Point<dim>>& points,
                                 std::vector<Vector<double>>& values) const;
  virtual void vector_values(const std::vector<Point<dim>>& points,
                             std::vector<std::vector<double>>& values) const;
};

template <int dim>
Startup<dim>::Startup()
  : Function<dim>(dim)
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
    values[k](0) = 1. * p(0);
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
    values[0][k] = 1. * p(0);
  }
}

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  AmandusParameters param;
  ::Elasticity::Parameters::declare_parameters(param);
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices) ;
  GridGenerator::hyper_cube(tr, -1, 1, true);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  Startup<d> startup;
  std::set<unsigned int> boundaries;
  boundaries.insert(0);
  boundaries.insert(1);

  ::Elasticity::Parameters parameters;
  parameters.parse_parameters(param);
  ::Elasticity::Matrix<d> matrix_integrator(parameters, boundaries);
  ::Elasticity::Residual<d> rhs_integrator(parameters, startup, boundaries);
  rhs_integrator.input_vector_names.push_back("Newton iterate");

  AmandusUMFPACK<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  app.set_boundary(1);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);

  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dofs());

  Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.parse_parameters(param);

  newton.initialize(newout);
  newton.debug_vectors = true;

  global_refinement_nonlinear_loop<d>(3, app, newton, 0, 0, &startup);
}
