#ifndef __darcy_checkerboard_solution
#define __darcy_checkerboard_solution

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <darcy/checkerboard/solution_parameters.h>

#include <cmath>
#include <utility>

namespace Darcy
{
  namespace Checkerboard
  {
    using namespace dealii;

    inline std::pair<double, double> polar_coords(const Point<2>& p) {
      double theta = std::atan2(p[1], p[0]);
      if(theta < 0.0) {
        theta += 2 * numbers::PI;
      }
      return std::pair<double, double>(hypot(p[0], p[1]), theta);
    }


    /**
     * A tensor valued function which is piecewise constant on each of the
     * four quadrants taking the values given in the constructor.
     * TODO: inverse
     */
    template <int dim>
      class CheckerboardTensorFunction : public TensorFunction<2, dim>
    {
      public:
        typedef typename TensorFunction<2, dim>::value_type value_type;
        CheckerboardTensorFunction(const std::vector<double>& parameters);
        ~CheckerboardTensorFunction();

        virtual value_type value(const Point<dim>& p) const;

        CheckerboardTensorFunction<dim>& inverse();
      private:
        CheckerboardTensorFunction(CheckerboardTensorFunction<dim>* parent,
                                   bool nocopyconstructor);
        std::vector<double> parameters;
        CheckerboardTensorFunction* inverse_ptr;
        bool is_inverse;
    };

    template <int dim>
      CheckerboardTensorFunction<dim>::CheckerboardTensorFunction(
          const std::vector<double>& parameters) :
        parameters(parameters),
        is_inverse(false)
    {
      inverse_ptr = new CheckerboardTensorFunction<dim>(this, true);
    }

    template <int dim>
      CheckerboardTensorFunction<dim>::CheckerboardTensorFunction(
          CheckerboardTensorFunction<dim>* parent,
          bool nocopyconstructor) :
        parameters(4),
        is_inverse(true)
    {
      inverse_ptr = parent;
      for(unsigned int i = 0; i < 4; ++i)
      {
        parameters[i] = 1.0 / parent->parameters[i];
      }
    }

    template <int dim>
      CheckerboardTensorFunction<dim>::~CheckerboardTensorFunction()
      {
        if(!is_inverse)
        {
          delete inverse_ptr;
        }
      }

    template <int dim>
      CheckerboardTensorFunction<dim>& CheckerboardTensorFunction<dim>::inverse()
      {
        return *inverse_ptr;
      }

    template <int dim>
      typename CheckerboardTensorFunction<dim>::value_type
      CheckerboardTensorFunction<dim>::value(const Point<dim>& p) const
      {
        Assert(dim == 2, ExcNotImplemented());
        Tensor<2, dim> identity;
        for(unsigned int i = 0; i < dim; ++i)
        {
          identity[i][i] = 1.0;
        }
        unsigned int quadrant;
        if(p(1) > 0.0) {
          if(p(0) > 0.0) {
            quadrant = 1;
          } else {
            quadrant = 2;
          }
        } else {
          if(p(0) > 0.0) {
            quadrant = 4;
          } else {
            quadrant = 3;
          }
        }

        return parameters[quadrant - 1] * identity;
      }


    /**
     * The scalar component of the exact solution to a divergence free
     * solution of Darcy's equation with a checkerboard coefficient as given
     * by the argument to the constructor.
     */
    class ScalarSolution : public Function<2>
    {
      public:
        ScalarSolution(const std::vector<double>& parameters);

        virtual double value(const Point<2>& p, 
                             const unsigned int component = 0) const;
        virtual Tensor<1, 2> gradient(const Point<2>& p,
                                      const unsigned int component = 0) const;

      private:
        double mu(double theta) const;
        double dmu(double theta) const;

        SolutionParameters solution_parameters;
    };

    ScalarSolution::ScalarSolution(const std::vector<double>& parameters) :
      solution_parameters(parameters)
    {}

    double ScalarSolution::mu(double theta) const 
    { 
      double gamma = solution_parameters.parameters[0];
      double sigma = solution_parameters.parameters[1];
      double rho = solution_parameters.parameters[2];

      if(0 <= theta && theta <= numbers::PI/2.0)
      {
        // first quadrant
        return (std::cos((numbers::PI/2.0 - sigma) * gamma) *
                std::cos((theta - numbers::PI/2.0 + rho) * gamma));
      } else if(3 * numbers::PI/2.0 <= theta && theta < 2 * numbers::PI)
      {
        // fourth quadrant
        return (std::cos((numbers::PI/2.0 - rho) * gamma) *
                std::cos((theta - 3.0 * numbers::PI / 2.0 - sigma) * gamma));
      } else if(numbers::PI/2.0 < theta && theta <= numbers::PI)
      {
        // second quadrant
        return std::cos(rho * gamma) * std::cos((theta - numbers::PI +
                                                 sigma) * gamma);
      } else if(numbers::PI < theta && theta < 3 * numbers::PI/2.0)
      {
        // third quadrant
        return std::cos(sigma * gamma) * std::cos((theta - numbers::PI -
                                                   rho) * gamma);
      }

      Assert(false, ExcInternalError());
      return 0.0;
    }

    double ScalarSolution::dmu(double theta) const 
    { 
      double gamma = solution_parameters.parameters[0];
      double sigma = solution_parameters.parameters[1];
      double rho = solution_parameters.parameters[2];

      if(0 <= theta && theta <= numbers::PI/2.0)
      {
        // first quadrant
        return (-1.0) * gamma * std::cos((numbers::PI/2.0 - sigma) *
                                         gamma) * std::sin((theta -
                                                            numbers::PI/2.0
                                                            + rho) *
                                                           gamma);
      } else if(3 * numbers::PI/2.0 <= theta && theta < 2 * numbers::PI)
      {
        // fourth quadrant
        return ((-1.0) * gamma * std::cos((numbers::PI/2.0 - rho) * gamma)
                * std::sin((theta - 3.0 * numbers::PI/2.0 - sigma) * gamma));
      } else if(numbers::PI/2.0 < theta && theta <= numbers::PI)
      {
        // second quadrant
        return (-1.0) * gamma * std::cos(rho * gamma) * std::sin((theta -
                                                                  numbers::PI
                                                                  +
                                                                  sigma)
                                                                 *
                                                                 gamma);
      } else if(numbers::PI < theta && theta < 3 * numbers::PI/2.0)
      {
        // third quadrant
        return (-1.0) * gamma * std::cos(sigma * gamma) * std::sin((theta
                                                                    -
                                                                    numbers::PI
                                                                    -
                                                                    rho)
                                                                   *
                                                                   gamma);
      }

      Assert(false, ExcInternalError());
      return 0.0;
    }

    double ScalarSolution::value(const Point<2>& p, 
                                 const unsigned int component) const 
    { 
      double gamma = solution_parameters.parameters[0];

      std::pair<double, double> polar = polar_coords(p);
      double r = polar.first;
      double theta = polar.second;

      return std::pow(r, gamma) * mu(theta);
    }

    Tensor<1, 2> ScalarSolution::gradient(const Point<2>& p, 
                                          const unsigned int component) const 
    { 
      const double gamma = solution_parameters.parameters[0];

      Tensor<1, 2> grad;
      std::pair<double, double> polar = polar_coords(p);
      double r = polar.first;
      double theta = polar.second;

      grad[0] = std::pow(r, gamma - 1.0) * (gamma * mu(theta) *
                                            std::cos(theta) -
                                            dmu(theta) *
                                            std::sin(theta));
      grad[1] = std::pow(r, gamma - 1.0) * (gamma * mu(theta) *
                                            std::sin(theta) +
                                            dmu(theta) *
                                            std::cos(theta));
      return grad;
    }



    /**
     * The exact mixed solution to a divergence free
     * solution of Darcy's equation with a checkerboard coefficient as given
     * by the argument to the constructor, i.e. if \f$p\f$ denotes the
     * exact scalar solution and \f$K\f$ the checkerboard coefficient
     * \f[
     * \left(
     * \begin{array}{c}
     * -K \nabla p \\
     *  p
     *  \f]
     */
    template <int dim>
      class MixedSolution : public Function<dim>
    {
      public:
        MixedSolution(const std::vector<double>& parameters);

        virtual double value(const Point<dim>& p,
                             const unsigned int component) const;

        /**
         * \brief The exact scalar solution.
         */
        const ScalarSolution scalar_solution;
        /**
         * \brief The corresponding checkerboard coefficient.
         */
        const CheckerboardTensorFunction<dim> coefficient;
    };

    template <int dim>
      MixedSolution<dim>::MixedSolution(const std::vector<double>& parameters) :
        Function<dim>(dim + 1), 
        scalar_solution(parameters), 
        coefficient(parameters)
    {}


    template <int dim>
      double MixedSolution<dim>::value(const Point<dim>& p,
                                       const unsigned int component) const
      {
        if(component == dim)
        {
          return scalar_solution.value(p);
        } else {
          Tensor<1, dim> grad_u = scalar_solution.gradient(p);
          Tensor<2, dim> coefficient_value = coefficient.value(p);

          Tensor<1, dim> value = -1.0 * coefficient_value * grad_u;
          return value[component];
        }
      }


    /*
       template <int dim>
       class Parameters
       {
       public:
       Parameters(const std::vector<double>& parameters);

       const std::vector<double> parameters;
       CheckerboardTensorFunction<dim> coefficient_tensor;
       CheckerboardTensorFunction<dim> inverse_coefficient_tensor;
       MixedSolution<dim> mixed_solution;

       private:
       std::vector<double> inverse_parameters(
       const std::vector<double>& parameters);
       };

       template <int dim>
       Parameters<dim>::Parameters(const std::vector<double>& parameters) :
       parameters(parameters),
       coefficient_tensor(parameters),
       inverse_coefficient_tensor(inverse_parameters(parameters)),
       mixed_solution(parameters)
       {}

       template <int dim>
       std::vector<double> Parameters<dim>::inverse_parameters(
       const std::vector<double>& parameters)
       {
       std::vector<double> inverse_parameters(parameters.size());
       for(unsigned int i = 0; i < parameters.size(); ++i)
       {
       inverse_parameters[i] = 1.0 / parameters[i];
       }

       return inverse_parameters;
       }
       */
  }
}

#endif
