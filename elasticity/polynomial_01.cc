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
#include <deal.II/base/function.h>
#include <apps.h>
#include <elasticity/parameters.h>
#include <elasticity/residual.h>
#include <elasticity/matrix.h>
#include <elasticity/polynomial.h>

#include <boost/scoped_ptr.hpp>

using namespace dealii;

// Keeping this dealii::Function class here for consistency, but
// leaving all values equal to zero.
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
  const std::vector<Point<dim> > &,
  std::vector<Vector<double> >   &) const
{
}

  
template <int dim>
void
Startup<dim>::vector_values (
  const std::vector<Point<dim> > &,
  std::vector<std::vector<double> >   &) const
{
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
  GridGenerator::hyper_cube (tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();

  Polynomials::Polynomial<double> grad_potential;
  grad_potential += Polynomials::Monomial<double>(4, 1.);
  grad_potential += Polynomials::Monomial<double>(2, -2.);
  grad_potential += Polynomials::Monomial<double>(0, 1.);
  grad_potential.print(std::cout);
  Polynomials::Polynomial<double> null;
  null += Polynomials::Monomial<double>(0, 0.);

  std::vector<Polynomials::Polynomial<double> > potentials(2);
  potentials[0] = grad_potential;
  potentials[1] = null;
  
  Startup<d> startup;
  
  ::Elasticity::Parameters parameters;
  parameters.parse_parameters(param);
  ::Elasticity::Matrix<d> matrix_integrator(parameters);
  ::Elasticity::PolynomialRHS<d> rhs_integrator(parameters, potentials);
  ::Elasticity::PolynomialError<d> error_integrator(parameters, potentials);
  
  AmandusUMFPACK<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  AmandusSolve<d>       solver(app, matrix_integrator);
  AmandusResidual<d>    residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);
  
  global_refinement_linear_loop(5, app, solver, residual, &error_integrator);
}
