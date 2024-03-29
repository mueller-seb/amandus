/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __tests_h
#define __tests_h

#include <deal.II/algorithms/any_data.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>

#include <amandus/amandus.h>

/**
 * Error is not close to machine accuracy
 */
DeclException1(ExcErrorTooLarge, double, << "error " << arg1 << " is too large");

/**
 * This function solves an equation on a given mesh and computes the
 * errors.
 *
 * The errors per cell are returned in the BlockVector used as first
 * argument. The number of blocks has to be at least the number of
 * errors computed by the error integrator, but the blocks may be
 * empty. They will be resized.
 *
 * @ingroup Verification
 */
template <int dim>
void
solve_and_error(dealii::BlockVector<double>& errors, AmandusApplicationSparse<dim>& app,
                dealii::Algorithms::OperatorBase& solver,
                dealii::Algorithms::OperatorBase& residual, const AmandusIntegrator<dim>& error)
{
  dealii::Vector<double> res;
  dealii::Vector<double> sol;

  solver.notify(dealii::Algorithms::Events::remesh);
  app.setup_system();
  app.setup_vector(res);
  app.setup_vector(sol);

  dealii::AnyData solution_data;
  dealii::Vector<double>* p = &sol;
  solution_data.add(p, "solution");

  dealii::AnyData data;
  dealii::Vector<double>* rhs = &res;
  data.add(rhs, "RHS");
  dealii::AnyData residual_data;
  residual(data, residual_data);
  dealii::deallog << "Residual " << res.l2_norm() << std::endl;
  solver(solution_data, data);
  app.output_results(0, &solution_data);

  app.error(errors, solution_data, error);
}

/**
 * This function solves an equation with an iterative solver on a given
 * mesh and computes the errors.
 *
 * The BlockVector will be resized according to the number of errors
 * computed by the ErrorIntegrator. If a initial_function is provided, it
 * will be projected to the Finite Element space and used as a starting
 * value for the iterative solver.
 *
 * @ingroup Verification
 */
template <int dim>
void
iterative_solve_and_error(dealii::BlockVector<double>& errors, AmandusApplicationSparse<dim>& app,
                          dealii::Algorithms::OperatorBase& solver,
                          const ErrorIntegrator<dim>& error,
                          const dealii::Function<dim>* initial_function = 0,
                          unsigned int n_qpoints = 0)
{
  dealii::Vector<double> solution;

  app.setup_system();
  app.setup_vector(solution);
  if (n_qpoints == 0)
  {
    n_qpoints = app.dofs().get_fe().tensor_degree() + 1;
  }
  dealii::QGauss<dim> quadrature(n_qpoints);
  if (initial_function != 0)
  {
    dealii::VectorTools::project(
      app.dofs(), app.hanging_nodes(), quadrature, *initial_function, solution);
  }

  solver.notify(dealii::Algorithms::Events::remesh);

  dealii::AnyData solution_data;
  dealii::Vector<double>* p = &solution;
  solution_data.add(p, "solution");

  dealii::AnyData data;
  solver(solution_data, data);
  app.error(errors, solution_data, error);
}

/**
 * A residual operator which represents a given residual operator minus this
 * residual operator applied to the exact solution. I.e. if an
 * AmandusResidual represents the action
 * \f[
 * u \mapsto F(u)
 * \f]
 * for the given integrator, then ExactResidual represents the action
 * \f[
 * u \mapsto F(u) - F(u_0)
 * \f]
 * where \f$u_0\f$ is the exact_solution.
 *
 * This is useful for verifying that we can recover a given function from a
 * combination of system integrator, residual integrator and linear solver
 * with Newton's method (notice that the derivative of the new objective
 * does not change, thus it is sufficient to adjust the residual operator
 * used in Newton's method without changing the inverse_derivative operator).
 *
 * @ingroup Verification
 */
template <int dim>
class ExactResidual : public AmandusResidual<dim>
{
public:
  ExactResidual(const AmandusApplicationSparse<dim>& application,
                AmandusIntegrator<dim>& integrator, const dealii::Function<dim>& exact_solution,
                unsigned int n_qpoints)
    : AmandusResidual<dim>(application, integrator)
    , exact_solution(&exact_solution)
    , quadrature(n_qpoints)
  {
  }

  virtual void operator()(dealii::AnyData& out, const dealii::AnyData& in) override
  {
    dealii::LogStream::Prefix p("ExactResidual");

    if (this->notifications.test(dealii::Algorithms::Events::initial) ||
        this->notifications.test(dealii::Algorithms::Events::remesh))
    {
      dealii::deallog << "Projecting exact solution." << std::endl;
      this->application->setup_vector(projection);
      dealii::VectorTools::project(this->application->dofs(),
                                   this->application->hanging_nodes(),
                                   quadrature,
                                   *exact_solution,
                                   projection);
      this->notifications.clear();
    }

    AmandusResidual<dim>::operator()(out, in);

    dealii::AnyData exact_residual_out;
    dealii::AnyData exact_residual_in;

    dealii::Vector<double> exact_residual;
    this->application->setup_vector(exact_residual);

    // calculate again a residual, but this time with the exact solution
    // as input. Notice that this works only because AnyData uses the
    // _first_ entry it finds for a given name.
    exact_residual_in.add<const dealii::Vector<double>*>(&projection, "Newton iterate");
    exact_residual_in.merge(in);
    exact_residual_out.add<dealii::Vector<double>*>(&exact_residual, "Residual");
    AmandusResidual<dim>::operator()(exact_residual_out, exact_residual_in);

    // subtract the residual of the exact solution
    out.entry<dealii::Vector<double>*>("Residual")->add(-1.0, exact_residual);
  }

private:
  const dealii::Function<dim>* exact_solution;
  const dealii::QGauss<dim> quadrature;
  dealii::Vector<double> projection;
};

/**
 * Simple class representing the dim-dimensional product of the given
 * one-dimensional polynomial pol. Useful in conjunction with ExactResidual
 * to verify the exact recovery for different orders of the Finite Element
 * in different dimensions.
 *
 * @ingroup Verification
 */
template <int dim>
class TensorProductPolynomial : public dealii::Function<dim>
{
public:
  TensorProductPolynomial(const dealii::Polynomials::Polynomial<double>& pol,
                          unsigned int n_components = 1)
    : dealii::Function<dim>(n_components)
    , polynomial(&pol)
    , derivative(pol.derivative())
    , derivative2(derivative.derivative())
    , derivative3(derivative2.derivative())
  {
  }

  virtual double
  value(const dealii::Point<dim>& p, const unsigned int /*component = 0*/) const override
  {
    double value = 1;
    for (std::size_t d = 0; d < dim; ++d)
    {
      value *= polynomial->value(p(d));
    }
    return value;
  }

  virtual dealii::Tensor<1, dim>
  gradient(const dealii::Point<dim>& p, const unsigned int /*component = 0*/) const override
  {
    dealii::Tensor<1, dim> grad;
    for (std::size_t d = 0; d < dim; ++d)
    {
      grad[d] = 1.0;
      for (std::size_t i = 0; i < dim; ++i)
      {
        grad[d] *= (i != d) ? polynomial->value(p(i)) : derivative.value(p(d));
      }
    }

    return grad;
  }

  virtual double
  laplacian(const dealii::Point<dim>& p, const unsigned int /*component = 0*/) const override
  {
    double laplace = 0.0;
    for (std::size_t d = 0; d < dim; ++d)
    {
      double factor = 1.0;
      for (std::size_t i = 0; i < dim; ++i)
      {
        factor *= (i != d) ? polynomial->value(p(i)) : derivative2.value(p(d));
      }
      laplace += factor;
    }
    return laplace;
  }

  dealii::Tensor<1, dim>
  gradient_laplacian(const dealii::Point<dim>& p, const unsigned int /*component = 0*/) const
  {
    dealii::Tensor<1, dim> grad;
    for (std::size_t d = 0; d < dim; ++d)
    {
      double grad1 = 1.0;
      double grad2 = 0.0;
      for (std::size_t i = 0; i < dim; ++i)
      {
        grad1 *= (i != d) ? polynomial->value(p(i)) : derivative3.value(p(d));
      }
      for (std::size_t j = 0; j < dim; ++j)
      {
        if (j != d)
        {
          double grad2factor = 1.0;
          for (std::size_t i = 0; i < dim; ++i)
          {
            if (i != j && i != d)
            {
              grad2factor *= polynomial->value(p(i));
            }
            else if (i == d)
            {
              grad2factor *= derivative.value(p(i));
            }
            else
            {
              grad2factor *= derivative2.value(p(i));
            }
          }
          grad2 += grad2factor;
        }
      }
      grad[d] = grad1 + grad2;
    }

    return grad;
  }

private:
  const dealii::Polynomials::Polynomial<double>* polynomial;
  const dealii::Polynomials::Polynomial<double> derivative;
  const dealii::Polynomials::Polynomial<double> derivative2;
  const dealii::Polynomials::Polynomial<double> derivative3;
};

/**
 * @ingroup Verification
 */
template <int dim>
void
verify_residual(unsigned int n_refinements, AmandusApplicationSparse<dim>& app,
                AmandusIntegrator<dim>& matrix_integrator,
                AmandusIntegrator<dim>& residual_integrator)
{
  dealii::Vector<double> seed;
  dealii::Vector<double> diff;

  for (unsigned int s = 0; s < n_refinements; ++s)
  {
    app.refine_mesh(true);

    app.setup_system();
    app.setup_vector(seed);
    app.setup_vector(diff);

    for (unsigned int i = 0; i < seed.size(); ++i)
      seed(i) = dealii::Utilities::generate_normal_random_number(0., 1.);

    dealii::AnyData diff_data;
    dealii::Vector<double>* p = &diff;
    diff_data.add(p, "diff");

    dealii::AnyData data;
    dealii::Vector<double>* rhs = &seed;
    data.add(rhs, "Newton iterate");

    app.assemble_matrix(data, matrix_integrator);
    app.verify_residual(diff_data, data, residual_integrator);
    app.output_results(s, &diff_data);

    dealii::deallog << "Difference " << diff.l2_norm() << std::endl;
  }
}

/**
 * @ingroup Verification
 */
template <int dim>
void
verify_theta_residual(unsigned int n_refinements, AmandusApplicationSparse<dim>& app,
                      AmandusIntegrator<dim>& matrix_integrator,
                      AmandusIntegrator<dim>& residual_integrator)
{
  dealii::Vector<double> seed;
  dealii::Vector<double> prev;
  dealii::Vector<double> diff;

  for (unsigned int s = 0; s < n_refinements; ++s)
  {
    app.refine_mesh(true);

    app.setup_system();
    app.setup_vector(seed);
    app.setup_vector(prev);
    app.setup_vector(diff);

    for (unsigned int i = 0; i < seed.size(); ++i)
      seed(i) = dealii::Utilities::generate_normal_random_number(0., 1.);
    for (unsigned int i = 0; i < prev.size(); ++i)
      prev(i) = dealii::Utilities::generate_normal_random_number(0., 1.);

    double dt = .73;

    dealii::AnyData diff_data;
    diff_data.add(&diff, "diff");

    dealii::AnyData data;
    data.add(&dt, "Timestep");
    data.add(&seed, "Newton iterate");

    matrix_integrator.extract_data(data);
    residual_integrator.extract_data(data);

    app.assemble_matrix(data, matrix_integrator);
    app.verify_residual(diff_data, data, residual_integrator);
    app.output_results(s, &diff_data);

    dealii::deallog << "Difference " << diff.l2_norm() << std::endl;
  }
}

#endif
