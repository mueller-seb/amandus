/**
 * @file
 *
 * @brief Instationary Cahn-Hilliard model.
 * @ingroup Cahn_Hilliardgroup
 * 
 * The Cahn-Hilliard model, which in strong form reads
 * \f[
 * u' = - \Delta (\frac{1}{\epsilon} w'(u) - \epsilon \Delta u)
 * \f]
 * Where \f$w\f$ is usually a double-well energy function like
 * \f$(u^2 - 1)^2\f$. The model can be derived as a gradient flow,
 * minimizing the same Energy functional as the Allen-Cahn model. However,
 * unlike the Allen-Cahn, this model is mass conserving.
 *
 * It can be used as a model for seperated phases with \f$\epsilon\f$
 * controlling the width of the interface. Currently, multigrid is only
 * working for very large \f$\epsilon\f$ which is probably related to the fact
 * that the coarse grids can not resolve sharp interfaces.
 *
 * As the interface is the only interesting part of the solution, we can use
 * adaptive mesh refinement to calculate with relatively few degrees of
 * freedom.
 */ #
#include <boost/scoped_ptr.hpp>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>

#include <amandus/adaptivity.h>
#include <amandus/apps.h>

#include <amandus/cahn_hilliard/massout.h>
#include <amandus/cahn_hilliard/matrix.h>
#include <amandus/cahn_hilliard/residual.h>
#include <amandus/cahn_hilliard/samples.h>

  using namespace dealii;

template <int dim>
class RefineStrategyCahnHillard
{
public:
  RefineStrategyCahnHillard(double refine_threshold, double coarsen_threshold)
    : refine_threshold(refine_threshold)
    , coarsen_threshold(coarsen_threshold)
  {
  }

  void operator()(Triangulation<dim>& tria, const BlockVector<double>& indicator)
  {
    GridRefinement::refine(tria, indicator.block(0), this->refine_threshold);
    GridRefinement::coarsen(tria, indicator.block(0), this->coarsen_threshold);
  }

protected:
  double refine_threshold, coarsen_threshold;
};

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  AmandusParameters param;

  param.enter_subsection("Model");
  param.declare_entry("Startup", "1");
  param.declare_entry("Diffusion", "0.1");
  param.declare_entry("AdvectionStrength", "0.0");
  param.declare_entry("RefineThreshold", "1.0");
  param.declare_entry("CoarsenThreshold", "1.0");
  param.declare_entry("InitialAdaption", "9");
  param.leave_subsection();

  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Model");
  int startup_no = param.get_integer("Startup");
  double diffusion = param.get_double("Diffusion");
  double advectionstrength = param.get_double("AdvectionStrength");
  double threshold = param.get_double("RefineThreshold");
  double c_threshold = param.get_double("CoarsenThreshold");
  unsigned int refine_loops = param.get_double("InitialAdaption");
  Function<d>* advection = CahnHilliard::advectionselector<d>(0, advectionstrength);
  param.leave_subsection();

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  std::vector<bool> mask;
  mask.push_back(false);
  mask.push_back(true);
  BlockMask timemask(mask);

  CahnHilliard::Matrix<d> matrix_stationary(diffusion, *advection);
  Integrators::Theta<d> matrix_integrator(matrix_stationary, true, timemask);
  matrix_integrator.input_vector_names.push_back("Newton iterate");
  CahnHilliard::Residual<d> residual_integrator(diffusion, *advection);
  Integrators::Theta<d> explicit_integrator(residual_integrator, false, timemask, true);
  explicit_integrator.input_vector_names.push_back("Previous iterate");
  Integrators::Theta<d> implicit_integrator(residual_integrator, true, timemask);
  implicit_integrator.input_vector_names.push_back("Newton iterate");

  AmandusApplicationSparse<d> app(tr, *fe, true);
  // AmandusApplication<d> app(tr, *fe);   // only for large diffusion
  app.parse_parameters(param);
  app.setup_system();
  // app.set_meanvalue();
  AmandusResidual<d> expl(app, explicit_integrator);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, implicit_integrator);

  // Set up timestepping algorithm with embedded Newton solver

  // CahnHilliard::Remesher<Vector<double>, d> remesher(threshold, c_threshold);
  Integrators::H1ErrorIntegrator<d> h1_error_integrator;
  ZeroFunction<d> zero(2);
  ErrorIntegrator<d> refine_integrator(zero);
  ComponentMask refine_mask(2, false);
  refine_mask.set(1, true);
  refine_integrator.add(&h1_error_integrator, refine_mask);
  ErrorRemesher<Vector<double>, d> remesher(app, tr, refine_integrator);
  RefineStrategyCahnHillard<d> refine_strategy(threshold, c_threshold);
  remesher.flag_callback(refine_strategy);

  param.enter_subsection("Output");
  // Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  CahnHilliard::MassOutputOperator<Vector<double>, d> newout;
  newout.parse_parameters(param);
  newout.initialize(app.dofs());
  newout.initialize(&remesher);
  param.leave_subsection();

  Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.parse_parameters(param);
  newton.debug = 6;

  Algorithms::ThetaTimestepping<Vector<double>> timestepping(expl, newton);
  timestepping.set_output(newout);
  timestepping.parse_parameters(param);

  // Now we prepare for the actual timestepping

  timestepping.notify(dealii::Algorithms::Events::initial);
  tr.signals.post_refinement.connect([&timestepping]()
                                     {
                                       timestepping.notify(dealii::Algorithms::Events::remesh);
                                     });
  dealii::Vector<double> solution;
  app.setup_system();
  app.setup_vector(solution);

  Function<d>& startup = *CahnHilliard::selector<d>(startup_no);
  VectorTools::interpolate(app.dofs(), startup, solution);

  dealii::AnyData indata;
  indata.add(&solution, "solution");
  dealii::AnyData outdata;
  for (unsigned int i = 0; i < refine_loops; ++i)
  {
    remesher(indata, outdata);
    VectorTools::interpolate(app.dofs(), startup, solution);
  }
  timestepping(indata, outdata);
}
