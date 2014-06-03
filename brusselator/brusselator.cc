// $Id$

/**
 * @file
 *
 * @brief Instationary Brusselator problem
 * <ul>
 * <li>Instationary Brusselator model</li>
 * <li>Homogeneous Dirichlet boundary conditions</li>
 * <li>Exact polynomial solutionExact polynomial solution</li>
 * <li>Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <deal.II/base/function_lib.h>
#include <apps.h>
#include <brusselator/implicit.h>
#include <brusselator/explicit.h>
#include <brusselator/matrix.h>

template <int dim>
class Startup : public dealii::Function<dim>
{
  public:
    Startup();
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
				    std::vector<Vector<double> >   &values) const;
};


template <int dim>
Startup<dim>::Startup ()
		:
		Function<dim> (2)
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
      values[k](0) = 2. + .25*p(1);
      values[k](1) = 1. + .8*p(0);
    }
}


  
int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(2);
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, 0., 1.);
  tr.refine_global(4);
  
  const unsigned int degree = 1;
  FE_DGQ<d> feb(degree);
  FESystem<d> fe(feb, 2);

  Brusselator::Parameters parameters;
  parameters.alpha0 = .00;
  parameters.alpha1 = .00;
  parameters.A = 3.4;
  parameters.B = 1.;
  Brusselator::Matrix<d> matrix_integrator(parameters);
  Brusselator::ExplicitResidual<d> explicit_integrator(parameters);
  explicit_integrator.input_vector_names.push_back("Previous iterate");
  Brusselator::ImplicitResidual<d> implicit_integrator(parameters);
  implicit_integrator.input_vector_names.push_back("Newton iterate");

  AmandusApplicationSparseMultigrid<d> app(tr, fe);
  //AmandusUMFPACK<d> app(tr, fe);
  AmandusResidual<d> expl(app, explicit_integrator);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, implicit_integrator);

  // Set up timestepping algorithm with embedded Newton solver
  
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dof_handler);
  
  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.control.log_history(true);
  newton.control.set_reduction(1.e-14);
  newton.threshold(.2);
  
  Algorithms::ThetaTimestepping<Vector<double> > timestepping(expl, newton);
  timestepping.set_output(newout);
  timestepping.theta(0.5);
  timestepping.timestep_control().start_step(.1);
  timestepping.timestep_control().final(20.);

  // Now we prepare for the actual timestepping
  
  timestepping.notify(dealii::Algorithms::Events::remesh);
  dealii::Vector<double> solution;
  app.setup_system();
  app.setup_vector(solution);

  Startup<d> startup;
  VectorTools::interpolate(app.dof_handler, startup, solution);
  
  dealii::AnyData indata;
  indata.add(&solution, "solution");
  dealii::AnyData outdata;
  timestepping(indata, outdata);  
}
