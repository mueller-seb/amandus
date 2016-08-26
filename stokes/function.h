/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __stokes_function_h
#define __stokes_function_h

#include <deal.II/base/flow_function.h>

using namespace dealii;
using namespace Functions;

namespace StokesFlowFunction
{
template <int dim>
class StokesPolynomial : public FlowFunction<dim>
{
public:
  virtual void vector_values(const std::vector<Point<dim>>& points,
                             std::vector<std::vector<double>>& values) const;
  virtual void vector_gradients(const std::vector<Point<dim>>& points,
                                std::vector<std::vector<Tensor<1, dim>>>& gradients) const;
  virtual void vector_laplacians(const std::vector<Point<dim>>& points,
                                 std::vector<std::vector<double>>& values) const;
};

template <int dim>
void
StokesPolynomial<dim>::vector_values(const std::vector<Point<dim>>& points,
                                     std::vector<std::vector<double>>& values) const
{
  unsigned int n = points.size();

  Assert(values.size() == dim + 1, ExcDimensionMismatch(values.size(), dim + 1));
  for (unsigned int d = 0; d < dim + 1; ++d)
    Assert(values[d].size() == n, ExcDimensionMismatch(values[d].size(), n));

  for (unsigned int k = 0; k < n; ++k)
  {
    const Point<dim>& p = points[k];
    const double x = p(0);
    const double y = p(1);

    if (dim == 2)
    {
      values[0][k] = 3. * x * y * y - x * x * x;
      values[1][k] = 3. * x * x * y - y * y * y;
      values[2][k] = 0. + this->mean_pressure;
    }
    else
    {
      Assert(false, ExcNotImplemented());
    }
  }
}

template <int dim>
void
StokesPolynomial<dim>::vector_gradients(const std::vector<Point<dim>>& points,
                                        std::vector<std::vector<Tensor<1, dim>>>& values) const
{
  unsigned int n = points.size();

  Assert(values.size() == dim + 1, ExcDimensionMismatch(values.size(), dim + 1));
  for (unsigned int d = 0; d < dim + 1; ++d)
    Assert(values[d].size() == n, ExcDimensionMismatch(values[d].size(), n));

  for (unsigned int k = 0; k < n; ++k)
  {
    const Point<dim>& p = points[k];
    const double x = p(0);
    const double y = p(1);

    if (dim == 2)
    {
      values[0][k][0] = 3. * y * y - 3. * x * x;
      values[0][k][1] = 6. * x * y;
      values[1][k][0] = 6. * x * y;
      values[1][k][1] = 3. * x * x - 3. * y * y;
      values[2][k][0] = 0.;
      values[2][k][1] = 0.;
    }
    else
    {
      Assert(false, ExcNotImplemented());
    }
  }
}

template <int dim>
void
StokesPolynomial<dim>::vector_laplacians(const std::vector<Point<dim>>& points,
                                         std::vector<std::vector<double>>& values) const
{
  unsigned int n = points.size();

  Assert(values.size() == dim + 1, ExcDimensionMismatch(values.size(), dim + 1));
  for (unsigned int d = 0; d < dim + 1; ++d)
    Assert(values[d].size() == n, ExcDimensionMismatch(values[d].size(), n));

  for (unsigned int k = 0; k < n; ++k)
  {
    const Point<dim>& p = points[k];
    const double x = p(0);
    const double y = p(1);

    if (dim == 2)
    {
      values[0][k] = 0.;
      values[1][k] = 0.;
      values[2][k] = 0.;
    }
    else
    {
      Assert(false, ExcNotImplemented());
    }
  }
}
}

#endif
