/**
 * @file
 * <ul>
 * <li> Stationary linear elasticity equations</li>
 * <li> Homogeneous Dirichlet boundary condition</li>
 * <li> Exact polynomial solution</li>
 * <li> Multigrid preconditioner with Schwarz-smoother</li>
 * </ul>
 *
 * @ingroup Elasticitygroup
 */

#include <amandus/apps.h>
#include <amandus/elasticity/matrix.h>
#include <amandus/elasticity/polynomial.h>
#include <amandus/elasticity/residual.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <elasticity/parameters.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;

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
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();
  std::set<unsigned int> boundaries;
  boundaries.insert(0);

  Polynomials::Polynomial<double> grad_potential;
  grad_potential += Polynomials::Monomial<double>(4, 1.);
  grad_potential += Polynomials::Monomial<double>(2, -2.);
  grad_potential += Polynomials::Monomial<double>(0, 1.);
  grad_potential.print(std::cout);
  Polynomials::Polynomial<double> curl_potential = grad_potential;
  grad_potential *= 7.;
  curl_potential *= 3.;

  std::vector<Polynomials::Polynomial<double>> potentials(2);
  potentials[0] = grad_potential;
  potentials[1] = curl_potential;

  ::Elasticity::Parameters parameters;
  parameters.parse_parameters(param);
  ::Elasticity::Matrix<d> matrix_integrator(parameters, boundaries);
  ::Elasticity::PolynomialRHS<d> rhs_integrator(parameters, potentials);
  ::Elasticity::PolynomialError<d> error_integrator(parameters, potentials);
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

  global_refinement_linear_loop(5, app, solver, residual, &error_integrator);
}
