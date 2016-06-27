/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __darcy_checkerboard_constrained_newton_h
#define __darcy_checkerboard_constrained_newton_h

#include <deal.II/algorithms/newton.h>

#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector_memory.h>

namespace Darcy
{
namespace Checkerboard
{
using namespace dealii;
using namespace Algorithms;

/**
 * This class is just a helper class to calculate the parameters for the
 * exact solution of Darcy's problem with a checkerboard diffusion
 * pattern. It is not intended for use in applications and might be
 * replaced by a more specialized implementation for this task in the
 * future.
 **/
template <class VECTOR>
class ConstrainedNewton : public Newton<VECTOR>
{
public:
  ConstrainedNewton(OperatorBase& residual, OperatorBase& inverse_derivative);
  virtual void operator()(AnyData& out, const AnyData& in);

  bool constraint_violation(const VECTOR& u);

private:
  SmartPointer<OperatorBase, Newton<VECTOR>> residual;

  SmartPointer<OperatorBase, Newton<VECTOR>> inverse_derivative;

  const unsigned int n_stepsize_iterations;
};

template <class VECTOR>
ConstrainedNewton<VECTOR>::ConstrainedNewton(OperatorBase& residual,
                                             OperatorBase& inverse_derivative)
  : Newton<VECTOR>(residual, inverse_derivative)
  , residual(&residual)
  , inverse_derivative(&inverse_derivative)
  , n_stepsize_iterations(21)
{
}

template <class VECTOR>
void
ConstrainedNewton<VECTOR>::operator()(AnyData& out, const AnyData& in)
{
  Assert(out.size() == 1, ExcNotImplemented());
  deallog.push("Newton");

  VECTOR& u = *out.entry<VECTOR*>(0);

  GrowingVectorMemory<VECTOR> mem;
  typename VectorMemory<VECTOR>::Pointer Du(mem);
  typename VectorMemory<VECTOR>::Pointer res(mem);

  res->reinit(u);
  AnyData src1;
  AnyData src2;
  src1.add<const VECTOR*>(&u, "Newton iterate");
  src1.merge(in);
  src2.add<const VECTOR*>(res, "Newton residual");
  src2.merge(src1);
  AnyData out1;
  out1.add<VECTOR*>(res, "Residual");
  AnyData out2;
  out2.add<VECTOR*>(Du, "Update");

  unsigned int step = 0;
  // fill res with (f(u), v)
  (*residual)(out1, src1);
  double resnorm = res->l2_norm();
  double old_residual;

  while (this->control.check(step++, resnorm) == SolverControl::iterate)
  {
    Du->reinit(u);
    try
    {
      (*inverse_derivative)(out2, src2);
    }
    catch (SolverControl::NoConvergence& e)
    {
      deallog << "Inner iteration failed after " << e.last_step << " steps with residual "
              << e.last_residual << std::endl;
    }

    u.add(-1., *Du);
    old_residual = resnorm;
    (*residual)(out1, src1);
    resnorm = res->l2_norm();

    // Step size control
    unsigned int step_size = 0;
    while (resnorm >= old_residual || constraint_violation(u))
    {
      ++step_size;
      if (step_size > n_stepsize_iterations)
      {
        deallog << "No smaller stepsize allowed!";
        break;
      }
      if (this->control.log_history())
        deallog << "Trying step size: 1/" << (1 << step_size) << " since residual was " << resnorm
                << std::endl;
      u.add(1. / (1 << step_size), *Du);
      (*residual)(out1, src1);
      resnorm = res->l2_norm();
    }
  }
  deallog.pop();

  // in case of failure: throw exception
  if (this->control.last_check() != SolverControl::success)
    AssertThrow(
      false, SolverControl::NoConvergence(this->control.last_step(), this->control.last_value()));
  // otherwise exit as normal
}

template <class VECTOR>
bool
ConstrainedNewton<VECTOR>::constraint_violation(const VECTOR& u)
{
  double gamma = u[0];
  double sigma = u[1];
  double rho = u[2];
  if (!(0.0 < gamma && gamma < 2))
  {
    return true;
  }
  if (!(std::max(0.0, numbers::PI * gamma - numbers::PI) < 2 * rho * gamma &&
        2 * rho * gamma < std::min(numbers::PI * gamma, numbers::PI)))
  {
    return true;
  }
  if (!(std::max(0.0, numbers::PI - numbers::PI * gamma) < -2 * sigma * gamma &&
        -2 * sigma * gamma < std::min(numbers::PI, 2 * numbers::PI - numbers::PI * gamma)))
  {
    return true;
  }
  return false;
}
}
}

#endif
