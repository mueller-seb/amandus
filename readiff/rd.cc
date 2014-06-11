// $Id$

/**
 * @file
 *
 * @brief Instationary ReactionDiffusion problem
 * <ul>
 * <li>Instationary ReactionDiffusion model</li>
 * <li>Homogeneous Dirichlet boundary conditions</li>
 * <li>Exact polynomial solutionExact polynomial solution</li>
 * <li>Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */

#include <deal.II/fe/fe_tools.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <deal.II/base/function.h>
#include <apps.h>
#include <readiff/residual.h>
#include <readiff/matrix.h>

#include <boost/scoped_ptr.hpp>

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


  
int main(int argc, const char** argv)
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(2);
  
  AmandusParameters param;
  ReactionDiffusion::Parameters::declare_parameters(param);
  param.read(argc, argv);
  param.log_parameters(deallog);
  
  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d> > fe(FETools::get_fe_from_name<d>(param.get("FE")));
  
  Triangulation<d> tr;
  GridGenerator::hyper_cube (tr, 0., 1.);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();
  
  ReactionDiffusion::Parameters parameters;
  parameters.parse_parameters(param);
  
  ReactionDiffusion::Matrix<d> m_integrator(parameters);
  Integrators::Theta<d> matrix_integrator(m_integrator, true);
  matrix_integrator.input_vector_names.push_back("Newton iterate");
  ReactionDiffusion::Residual<d> r_integrator(parameters);
  Integrators::Theta<d> explicit_integrator(r_integrator, false);
  explicit_integrator.input_vector_names.push_back("Previous iterate");
  Integrators::Theta<d> implicit_integrator(r_integrator, true);
  implicit_integrator.input_vector_names.push_back("Newton iterate");


  AmandusApplicationSparseMultigrid<d> app(tr, *fe);
  //AmandusUMFPACK<d> app(tr, *fe);
  AmandusResidual<d> expl(app, explicit_integrator);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, implicit_integrator);

  // Set up timestepping algorithm with embedded Newton solver
  
  param.enter_subsection("Output");
  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.parse_parameters(param);
  newout.initialize(app.dof_handler);
  param.leave_subsection();
  
  param.enter_subsection("Newton");
  Algorithms::Newton<Vector<double> > newton(residual, solver);
  newton.parse_parameters(param);
  param.leave_subsection();
  
  param.enter_subsection("ThetaTimestepping");
  Algorithms::ThetaTimestepping<Vector<double> > timestepping(expl, newton);
  timestepping.set_output(newout);
  timestepping.parse_parameters(param);
  param.leave_subsection();
  
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
