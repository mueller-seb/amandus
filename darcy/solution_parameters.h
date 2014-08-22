#ifndef __solution_parameters_h
#define __solution_parameters_h

#include <deal.II/base/function.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/algorithms/any_data.h>
#include <deal.II/base/parameter_handler.h>

#include "constrained_newton.h"

namespace DarcyCoefficient
{
  using namespace dealii;

  /**
   * A function which describes the parameters for the exact solution as its
   * root.
   */
  class ParameterEquation : public Function<3>
  {
    public:
      /**
       * Constructor.
       * @param[in] coefficient Value of the coefficient in the quadrants.
       */
      ParameterEquation(const std::vector<double>& coefficient);

      virtual double value(const Point<3>& p,
                           const unsigned int component) const;
      virtual Tensor<1, 3> gradient(const Point<3>& p,
                                    const unsigned int component) const;

    private:
      std::vector<double> rhs;
  };

  ParameterEquation::ParameterEquation(const std::vector<double>& coefficient) :
    Function<3>(3)
  {
    rhs.push_back(coefficient[0]/coefficient[1]);
    rhs.push_back(coefficient[1]/coefficient[2]);
    rhs.push_back(coefficient[2]/coefficient[3]);
  }

  double ParameterEquation::value(const Point<3>& p,
                                  const unsigned int component) const
  {
    double gamma = p[0];
    double sigma = p[1];
    double rho = p[2];

    double value = 0.0;

    if(component == 0)
    {
      value = std::tan((numbers::PI/2 - sigma) * gamma) *
        std::pow(std::tan(rho * gamma), -1);
    } else if(component == 1) {
      value = std::tan(rho * gamma) * std::pow(std::tan(sigma * gamma), -1);
    } else if(component == 2) {
      value = std::tan(sigma * gamma) * std::pow(std::tan((numbers::PI/2 -
                                                           rho) * gamma),
                                                 -1);
    } else {
      Assert(false, ExcInternalError());
    }

    return value + rhs[component];
  }

  Tensor<1, 3> ParameterEquation::gradient(const Point<3>& p,
                                           const unsigned int component) const
  {
    double gamma = p[0];
    double sigma = p[1];
    double rho = p[2];

    Tensor<1, 3> grad;

    if(component == 0)
    {
      grad[0] = (numbers::PI/2 - sigma) * std::pow(std::tan(rho * gamma),
                                                   -1) *
        std::pow(std::cos(numbers::PI/2 * gamma - sigma * gamma), -2) - rho
        * std::tan(numbers::PI/2 *gamma - sigma * gamma) *
        std::pow(std::tan(rho * gamma), -2) * std::pow(std::cos(rho *
                                                                gamma), -2);

      grad[1] = -1 * gamma * std::pow(std::tan(rho * gamma), -1) *
        std::pow(std::cos(numbers::PI/2 * gamma - sigma * gamma), -2);

      grad[2] = -1 * gamma * std::tan(numbers::PI/2 * gamma - sigma * gamma)
        * std::pow(std::cos(rho * gamma), -2) * std::pow(std::tan(rho *
                                                                  gamma),
                                                         -2);
    } else if(component == 1)
    {
      grad[0] = rho * std::pow(std::tan(sigma * gamma), -1) *
        std::pow(std::cos(rho * gamma), -2) - sigma * std::tan(rho * gamma)
        * std::pow(std::tan(sigma * gamma), -2) * std::pow(std::cos(sigma *
                                                                    gamma),
                                                           -2);

      grad[1] = -1 * gamma * std::tan(rho * gamma) * std::pow(std::cos(sigma
                                                                       *
                                                                       gamma),
                                                              -2) *
        std::pow(std::tan(sigma * gamma), -2);

      grad[2] = gamma * std::pow(std::cos(rho * gamma), -2) *
        std::pow(std::tan(sigma * gamma), -1);
    } else if(component == 2)
    {
      grad[0] = sigma * std::pow(std::cos(sigma * gamma), -2) *
        std::pow(std::tan(numbers::PI/2 * gamma - rho * gamma), -1) -
        (numbers::PI/2 - rho) * std::tan(sigma * gamma) *
        std::pow(std::tan(numbers::PI/2 * gamma - rho * gamma), -2) *
        std::pow(std::cos(numbers::PI/2 * gamma - rho * gamma), -2);

      grad[1] = gamma * std::pow(std::tan(numbers::PI/2 * gamma - rho *
                                          gamma), -1) *
        std::pow(std::cos(sigma * gamma), -2);

      grad[2] = gamma * std::tan(sigma * gamma) *
        std::pow(std::cos(numbers::PI/2 * gamma - rho * gamma), -2) *
        std::pow(std::tan(numbers::PI/2 * gamma - rho * gamma), -2);
    } else
    {
      Assert(false, ExcInternalError());
    }

    return grad;
  }


  /**
   * Convert a generic vector to a Point.
   */
  template<int dim, typename VECTOR>
    inline void pointify(const VECTOR& v,
                         Point<dim>& p)
    {
      AssertDimension(v.size(), dim);
      for(unsigned int i = 0; i < dim; ++i) 
      {
        p[i] = v[i];
      }
    }


  /**
   * Provides the Jacobian of a function as a FullMatrix.
   */
  template<int dim, typename number>
    inline void jacobian(const Function<dim>& f, 
                         const Point<dim>& p,
                         FullMatrix<number>& J)
    {
      AssertDimension(J.m(), f.n_components);
      AssertDimension(J.n(), dim);
      std::vector<Tensor<1, dim> > grads(f.n_components);
      f.vector_gradient(p, grads);
      for(unsigned int i = 0; i < f.n_components; ++i) 
      {
        for(unsigned int j = 0; j < dim; ++j)
        {
          J.set(i, j, grads[i][j]);
        }
      }
    }


  /** 
   * Wrapper for a function to provide the corresponding residual for
   * Newton's method.
   */
  template<int dim, typename VECTOR>
    class NewtonResidual : public Algorithms::Operator<VECTOR> 
  {
    public:
      NewtonResidual(const Function<dim>* f);
      virtual void operator()(AnyData& out, 
                              const AnyData& in);
    private:
      SmartPointer<const Function<dim> > f;
      Point<dim> p;
  };

  template<int dim, typename VECTOR>
    NewtonResidual<dim, VECTOR>::NewtonResidual(const Function<dim>* f) :
      f(f) {}

  template<int dim, typename VECTOR>
    void NewtonResidual<dim, VECTOR>::operator()(AnyData& out,
                                                 const AnyData& in)
    {
      Assert(in.size() == 1, ExcNotImplemented());
      Assert(out.size() == 1, ExcNotImplemented());

      pointify(*(in.read<const VECTOR*>(0)), p);
      f->vector_value(p, *out.entry<VECTOR*>(0));
    }


  /**
   * Wrapper for a function with implementented gradient to supply Newton's
   * method with the inverse Jacobian.
   */
  template<int dim, typename VECTOR>
    class NewtonInvJ : public Algorithms::Operator<VECTOR>
  {
    public:
      typedef typename VECTOR::value_type number;

      NewtonInvJ(const Function<dim>* f);
      virtual void operator()(AnyData& out,
                              const AnyData& in);
    private:
      SmartPointer<const Function<dim> > f;
      Point<dim> p;
      FullMatrix<number> J;
      FullMatrix<number> iJ;
  };

  template<int dim, typename VECTOR>
    NewtonInvJ<dim, VECTOR>::NewtonInvJ(const Function<dim>* f):
      f(f),
      J(dim, dim),
      iJ(dim, dim) {
        AssertDimension(dim, f->n_components);
      }

  template<int dim, typename VECTOR>
    void NewtonInvJ<dim, VECTOR>::operator()(AnyData& out,
                                             const AnyData& in)
    {
      Assert(out.size() == 1, ExcNotImplemented());
      Assert(in.size() == 2, ExcNotImplemented());
      Assert(in.name(0) == "Newton residual", ExcInternalError());
      Assert(in.name(1) == "Newton iterate", ExcInternalError());

      pointify(*(in.read<const VECTOR*>(1)), p);
      jacobian(*f, p, J);
      iJ.invert(J);
      iJ.vmult(*out.entry<VECTOR*>(0), *(in.read<const VECTOR*>(0)));
    }


  /**
   * Wraps the function into the Residual and Inverse Jacobian classes and
   * applies Newton's method.
   */
  template<int dim, typename VECTOR>
    void find_root(const Function<dim>& f,
                   VECTOR& u)
    {
      NewtonResidual<dim, VECTOR> res(&f);
      NewtonInvJ<dim, VECTOR> iJf(&f);
      ConstrainedNewton<VECTOR> newton(res, iJf);
      // TODO: consider exposing reduction control parameters
      ParameterHandler prm;
      newton.declare_parameters(prm);
      prm.enter_subsection("Newton");
      prm.set("Stepsize iterations", "10");
      prm.set("Tolerance", "1.e-12");
      prm.set("Reduction", "0.0");
      prm.set("Max steps", "100000");
      prm.set("Log history", false);
      prm.set("Log frequency", "10");
      prm.leave_subsection();
      newton.parse_parameters(prm);

      AnyData out;
      AnyData in;

      // make sure to have the pointer to u as an lvalue so out does not
      // get marked as being const
      VECTOR* p = &u;
      out.add(p, "Start and end value");
      newton(out, in);
    }


  /**
   * Simple interface class which provides the parameters of the exact
   * solution for a given coefficient.
   */
  class SolutionParameters
  {
    public:
      /**
       * Constructor calculates the desired parameters and stores them in
       * SolutionParameters::parameters.
       */
      SolutionParameters(const std::vector<double>& input_parameters);
      Vector<double> parameters; ///< The parameters of the exact solution.
  };

  SolutionParameters::SolutionParameters(
      const std::vector<double>& input_parameters) :
    parameters(3)
  {
    ParameterEquation parameter_equation(input_parameters);
    parameters[0] = 1.0;
    parameters[1] = -1 * dealii::numbers::PI/4.0;
    parameters[2] = numbers::PI/4.0;

    find_root(parameter_equation, parameters);
  }
}

#endif
