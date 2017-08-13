/**
 * @file
 * <code>polynomial_01.cc</code> with Newton solver
 * <ul>
 * <li> Instationary nonlinear biot equations (not yet implemented)</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Examples
 */

#include <amandus/apps.h>
#include <amandus/biot/matrix.h>
#include <amandus/biot/polynomial.h>
#include <biot/parameters.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;

int
main(int argc, const char** argv)
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(100);

  AmandusParameters param;
  ::Biot::Parameters::declare_parameters(param);
  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d>(param.get("FE")));

  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();
  std::set<unsigned int> boundaries;
  boundaries.insert(0);

  Polynomials::Polynomial<double> grad_potential;
  grad_potential += Polynomials::Monomial<double>(4, 1.);
  grad_potential += Polynomials::Monomial<double>(2, -2.);
  grad_potential += Polynomials::Monomial<double>(0, 1.);

  Polynomials::Polynomial<double> curl_potential = grad_potential;

  Polynomials::Polynomial<double> p_potential;
  p_potential += Polynomials::Monomial<double>(3, 1.);
  p_potential += Polynomials::Monomial<double>(1, -3.);

  //  grad_potential *= 7.;
  //  curl_potential *= 3.;

  std::vector<Polynomials::Polynomial<double>> potentials(10);
  potentials[0] = curl_potential;
  potentials[1] = curl_potential;
  potentials[2] = grad_potential;
  potentials[3] = grad_potential;
  potentials[4] = curl_potential;
  potentials[5] = curl_potential;
  potentials[6] = grad_potential;
  potentials[7] = grad_potential;
  potentials[8] = p_potential;
  potentials[9] = p_potential;

  ::Biot::Parameters parameters;
  parameters.parse_parameters(param);
  ::Biot::TestMatrix<d> matrix_integrator(parameters);
  ::Biot::PolynomialResidual<d> rhs_integrator(potentials, parameters);
  rhs_integrator.input_vector_names.push_back("Newton iterate");
  ::Biot::PolynomialError<d> error_integrator(potentials, parameters);
  if (fe->conforms(FiniteElementData<d>::H1))
  {
    matrix_integrator.use_boundary = false;
    matrix_integrator.use_face = false;
    rhs_integrator.use_boundary = false;
    rhs_integrator.use_face = false;
  }

  AmandusApplication<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);

  Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.parse_parameters(param);
  newton.debug = 3;

  global_refinement_nonlinear_loop(5, app, newton, &error_integrator);
}
