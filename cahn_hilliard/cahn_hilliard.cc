#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_tools.h>

#include <apps.h>

#include <cahn_hilliard/residual.h>
#include <cahn_hilliard/matrix.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;

template <int dim>
class Startup : public dealii::Function<dim>
{
  public:
    Startup();
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double>   &values,
                             const unsigned int component = 0) const;
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &values) const;
};


  template <int dim>
Startup<dim>::Startup ()
  :
    Function<dim>(2)
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
    if (std::fabs(p(0)) < .8 && std::fabs(p(1)) < .2)
      values[k](1) = 1.;
    else if (std::fabs(p(1)) < .8 && std::fabs(p(0)) < .2)
      values[k](1) = 1.;
    else
      values[k](1) = -1.;
  }
}


template <int dim>
void
Startup<dim>::value_list (
    const std::vector<Point<dim> > &points,
    std::vector<double>   &values,
    const unsigned int component) const
{
  AssertDimension(points.size(), values.size());

  if(component == 1)
  {
    for (unsigned int k=0;k<points.size();++k)
    {
      const Point<dim>& p = points[k];
      if (std::fabs(p(0)) < .8 && std::fabs(p(1)) < .2)
        values[k] = 1.;
      else if (std::fabs(p(1)) < .8 && std::fabs(p(0)) < .2)
        values[k] = 1.;
      else
        values[k] = -1.;
    }
  } else
  {
    for(unsigned int k = 0; k < points.size(); ++k)
    {
      values[k] = 0.0;
    }
  }
}


int main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(2);

  AmandusParameters param;
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d> > fe(FETools::get_fe_by_name<d, d>(param.get("FE")));
  /*
  FE_Q<d> first(1);
  FE_Q<d> second(2);
  FESystem<d> fe_system(first, 1
                        second, 1);
  const FiniteElement<d>* fe(&fe_system);
  */

  Triangulation<d> tr;
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  CahnHilliard::Matrix<d> matrix_stationary;
  Integrators::Theta<d> matrix_integrator(matrix_stationary, true);
  matrix_integrator.input_vector_names.push_back("Newton iterate");
  CahnHilliard::Residual<d> residual_integrator;
  Integrators::Theta<d> explicit_integrator(residual_integrator, false);
  explicit_integrator.input_vector_names.push_back("Previous iterate");
  Integrators::Theta<d> implicit_integrator(residual_integrator, true);
  implicit_integrator.input_vector_names.push_back("Newton iterate");

  AmandusApplicationSparse<d> app(tr, *fe);
  AmandusResidual<d> expl(app, explicit_integrator);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, implicit_integrator);

  // Set up timestepping algorithm with embedded Newton solver

  param.enter_subsection("Output");
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.parse_parameters(param);
  newout.initialize(app.dofs());
  param.leave_subsection();

  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.parse_parameters(param);
  newton.debug = 6;

  Algorithms::ThetaTimestepping<Vector<double> > timestepping(expl, newton);
  timestepping.set_output(newout);
  timestepping.parse_parameters(param);

  // Now we prepare for the actual timestepping

  timestepping.notify(dealii::Algorithms::Events::remesh);
  dealii::Vector<double> solution;
  app.setup_system();
  app.setup_vector(solution);

  Startup<d> startup;
  VectorTools::interpolate(app.dofs(), startup, solution);

  dealii::AnyData indata;
  indata.add(&solution, "solution");
  dealii::AnyData outdata;
  timestepping(indata, outdata);
}
