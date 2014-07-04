// $Id$

/**
 * @file
 * <ul>
 * <li> Stationary Elasticity equations</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Newton solver</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */

#include <deal.II/fe/fe_tools.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <apps.h>
#include <elasticity/parameters.h>
#include <elasticity/residual.h>
#include <elasticity/matrix.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;

template <int dim>
class Startup : public dealii::Function<dim>
{
  public:
    Startup();
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
				    std::vector<Vector<double> >   &values) const;
    virtual void vector_values (const std::vector<Point<dim> > &points,
				std::vector<std::vector<double> > & values) const;
};


template <int dim>
Startup<dim>::Startup ()
		:
		Function<dim> (dim)
{}


template <int dim>
void
Startup<dim>::vector_value_list (
  const std::vector<Point<dim> > &points,
  std::vector<Vector<double> >   &values) const
{
  AssertDimension(points.size(), values.size());
  
  for (unsigned int k=0;k<points.size();++k)
    {
      const Point<dim>& p = points[k];
      values[k](0) = .0001*p(0)*p(0)*p(0);	
    }
}

  
template <int dim>
void
Startup<dim>::vector_values (
  const std::vector<Point<dim> > &points,
  std::vector<std::vector<double> >   &values) const
{
  AssertVectorVectorDimension(values, this->n_components, points.size());
  
  for (unsigned int k=0;k<points.size();++k)
    {
      const Point<dim>& p = points[k];
      values[0][k] = .0001*p(0)*p(0)*p(0);	
    }
}

  
int main(int argc, const char** argv)
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  AmandusParameters param;
  ::Elasticity::Parameters::declare_parameters(param);
  param.read(argc, argv);
  param.log_parameters(deallog);
  
  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d> > fe(FETools::get_fe_from_name<d>(param.get("FE")));
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, -1, 1, true);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();
  
  ZeroFunction<d> startup(d);
  
  ::Elasticity::Parameters parameters;
  parameters.parse_parameters(param);
  ::Elasticity::Matrix<d> matrix_integrator(parameters);
  ::Elasticity::Residual<d> rhs_integrator(parameters, startup);
  rhs_integrator.input_vector_names.push_back("Newton iterate");
  
  AmandusUMFPACK<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  app.set_boundary(1);
  app.set_boundary(2);
  app.set_boundary(3);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d>    residual(app, rhs_integrator);
  
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dofs());

  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.parse_parameters(param);

  newton.initialize(newout);
  newton.debug_vectors = true;

  global_refinement_nonlinear_loop<d>(2, app, newton, 0, 0, &startup);
}
