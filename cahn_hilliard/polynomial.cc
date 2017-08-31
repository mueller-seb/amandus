/**
 * @file
 *
 * @brief Stationary, polynomial Cahn-Hilliard model
 *
 * <ul>
 * <li>Stationary Cahn-Hilliard model</li>
 * <li>Polynomial solution</li>
 * <li>Dimensions two and three</li>
 * <li>UMFPack and Multigrid preconditioner with Schwarz-smoother for large
 * diffusion</li>
 * </ul>
 *
 * @ingroup Verification
 */
#include <boost/scoped_ptr.hpp>
#include <deal.II/algorithms/newton.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>

#include <amandus/apps.h>
#include <amandus/tests.h>

#include <amandus/cahn_hilliard/matrix.h>
#include <amandus/cahn_hilliard/residual.h>

using namespace dealii;

template <int dim>
class CahnHilliardAdapter : public Function<dim>
{
public:
  CahnHilliardAdapter(const TensorProductPolynomial<dim>& tpp)
    : Function<dim>(2)
    , tpp(&tpp)
  {
  }

  virtual double
  value(const Point<dim>& p, const unsigned int component = 0) const
  {
    if (component == 0)
    {
      double value = tpp->value(p, 0);
      double laplacian = tpp->laplacian(p, 0);
      double eps = 10.0;

      return (value * (value * value - 1.0) / eps - eps * laplacian);
    }
    return tpp->value(p, 0);
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim>& p, const unsigned int component = 0) const
  {
    if (component == 0)
    {
      double eps = 10.0;
      dealii::Tensor<1, dim> minus_eps_grad_laplace_u =
        (-1.0 * eps * tpp->gradient_laplacian(p, 0));
      double value = tpp->value(p, 0);
      double factor = (3 * value * value - 1.0) / eps;
      dealii::Tensor<1, dim> factor_grad_u = factor * tpp->gradient(p, 0);

      return factor_grad_u + minus_eps_grad_laplace_u;
    }

    return tpp->gradient(p, 0);
  }

protected:
  const TensorProductPolynomial<dim>* tpp;
};

template <int dim>
class StartSine : public Function<dim>
{
public:
  StartSine()
    : Function<dim>(2)
  {
  }

  virtual double
  value(const Point<dim>& p, const unsigned int component = 0) const
  {
    double pihalf = numbers::PI / 2.0;
    double u = std::sin(pihalf * p(0)) * std::sin(pihalf * p(1));
    if (component == 1)
    {
      return u;
    }
    else
    {
      return -1.0 * pihalf * pihalf * u;
    }
  }

  virtual Tensor<1, dim>
  gradient(const Point<2>& p, const unsigned int component = 0) const
  {
    double pihalf = numbers::PI / 2.0;
    double u = std::sin(pihalf * p(0)) * std::sin(pihalf * p(1));

    dealii::Tensor<1, dim> grad;
    grad[0] = pihalf * std::cos(pihalf * p(0)) * std::sin(pihalf * p(1));
    grad[1] = pihalf * std::sin(pihalf * p(0)) * std::cos(pihalf * p(1));
    if (component == 1)
    {
      return grad;
    }
    else
    {
      return -1.0 * pihalf * pihalf * grad;
    }

    return grad;
  }
};

template <int dim>
class Start : public Function<dim>
{
public:
  Start()
    : Function<dim>(2)
  {
  }

  virtual double
  value(const Point<dim>& p, const unsigned int component = 0) const
  {
    double u = p(0) * p(1);
    if (component == 1)
    {
      return u;
    }
    else
    {
      return 0.0;
    }
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim>& p, const unsigned int component = 0) const
  {
    dealii::Tensor<1, dim> grad;
    if (component == 1)
    {
      grad[0] = p(1);
      grad[1] = p(0);
    }
    else
    {
      grad[0] = 0.0;
      grad[1] = 0.0;
    }

    return grad;
  }
};

template <int dim>
void
output_function(const Function<dim>& function, const DoFHandler<dim>& dofh)
{
  Vector<double> dofs;
  dofs.reinit(dofh.n_dofs());
  ConstraintMatrix no_constraints;
  no_constraints.close();
  QGauss<dim> quadrature(6);
  VectorTools::project(dofh, no_constraints, quadrature, function, dofs);
  DataOut<dim> data_out;
  data_out.add_data_vector(dofh, dofs, "projected_exact");
  data_out.build_patches(dofh.get_fe().tensor_degree());
  std::ofstream out_file("projected_exact.vtk");
  data_out.write_vtk(out_file);
}

template <int d>
void
run(AmandusParameters& param)
{
  param.enter_subsection("Model");
  double diffusion = param.get_double("Diffusion");
  param.leave_subsection();

  param.enter_subsection("Discretization");
  Triangulation<d> tr(Triangulation<d>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));

  boost::scoped_ptr<const FiniteElement<d>> fe(FETools::get_fe_by_name<d, d>(param.get("FE")));
  param.leave_subsection();

  Polynomials::Polynomial<double> solution1d;
  solution1d += Polynomials::Monomial<double>(1, 1.);
  // solution1d += Polynomials::Monomial<double>(1, -0.3);
  solution1d.print(std::cout);

  TensorProductPolynomial<d> tpp(solution1d);
  CahnHilliardAdapter<d> exact_solution(tpp);
  // TensorProductPolynomial<d> exact_solution(solution1d, 2);
  Start<d> newton_start;
  // StartSine<d> newton_start;

  std::vector<bool> mask;
  mask.push_back(false);
  mask.push_back(true);
  BlockMask timemask(mask);

  Functions::ZeroFunction<d> zero_advection(d);
  CahnHilliard::Matrix<d> matrix(diffusion, zero_advection);
  matrix.input_vector_names.push_back("Newton iterate");
  CahnHilliard::Residual<d> residual_integrator(diffusion, zero_advection);
  residual_integrator.input_vector_names.push_back("Newton iterate");

  Integrators::L2ErrorIntegrator<d> l2_error_integrator;
  Integrators::H1ErrorIntegrator<d> h1_error_integrator;
  std::vector<bool> othermask;
  othermask.push_back(true);
  othermask.push_back(false);
  ComponentMask errormask(othermask);
  ErrorIntegrator<d> error_integrator(exact_solution);
  // error_integrator.add(&l2_error_integrator, errormask);
  error_integrator.add(&h1_error_integrator, errormask);

  AmandusApplicationSparse<d>* app_init;
  param.enter_subsection("Testing");
  if (param.get_bool("Multigrid"))
  {
    app_init = new AmandusApplication<d>(tr, *fe);
  }
  else
  {
    app_init = new AmandusApplicationSparse<d>(tr, *fe, param.get_bool("UMFPack"));
  }
  param.leave_subsection();
  boost::scoped_ptr<AmandusApplicationSparse<d>> app(app_init);
  // app->set_meanvalue();
  app->parse_parameters(param);
  AmandusSolve<d> solver(*app, matrix);
  ExactResidual<d> residual(*app, residual_integrator, exact_solution, 12);
  // AmandusResidual<d> residual(*app, residual_integrator);

  param.enter_subsection("Output");
  Algorithms::DoFOutputOperator<Vector<double>, d> newton_output;
  newton_output.parse_parameters(param);
  newton_output.initialize(app->dofs());
  param.leave_subsection();

  Algorithms::Newton<Vector<double>> newton(residual, solver);
  newton.parse_parameters(param);
  newton.initialize(newton_output);
  newton.debug = 6;

  param.enter_subsection("Testing");
  int steps = param.get_integer("Number of global refinement loops");
  double TOL = param.get_double("Tolerance");
  param.leave_subsection();
  BlockVector<double> errors(2);
  double acc_error;
  for (int s = 0; s < steps; ++s)
  {
    app->setup_system();
    output_function<d>(exact_solution, app->dofs());
    iterative_solve_and_error<d>(errors, *app, newton, error_integrator, 0);
    //&newton_start);
    for (unsigned int i = 0; i < errors.n_blocks(); ++i)
    {
      acc_error = errors.block(i).l2_norm();
      deallog << "Error(" << i << "): " << acc_error << std::endl;
      Assert(acc_error < TOL, ExcErrorTooLarge(acc_error));
    }
    tr.refine_global(1);
  }
}

int
main(int argc, const char** argv)
{
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  deallog.depth_console(10);

  AmandusParameters param;

  param.enter_subsection("Testing");
  param.declare_entry("Number of global refinement loops", "3");
  param.declare_entry("Tolerance", "1.e-13");
  param.declare_entry("Multigrid", "true");
  param.declare_entry("UMFPack", "true");
  param.leave_subsection();

  param.enter_subsection("Model");
  param.declare_entry("Dimensionality", "2");
  param.declare_entry("Diffusion", "1.0");
  param.leave_subsection();

  param.read(argc, argv);
  param.log_parameters(deallog);

  param.enter_subsection("Model");
  const unsigned int d = param.get_integer("Dimensionality");
  param.leave_subsection();

  if (d == 3)
  {
    run<3>(param);
  }
  else
  {
    run<2>(param);
  }
}
