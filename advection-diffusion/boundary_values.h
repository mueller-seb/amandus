/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __advectiondiffusion_boundaryvalues_h
#define __advectiondiffusion_boundaryvalues_h

using namespace dealii;
using namespace LocalIntegrators;
using namespace MeshWorker;

/**
 * Computes the boundary values.
 *
 *
 * The boundary values are only evaluated at the inflow boundary!
 */

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues(){};
  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component = 0) const override;
};
template <int dim>
void
BoundaryValues<dim>::value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                                const unsigned int) const
{
  Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
  for (unsigned int i = 0; i < values.size(); ++i)
  {
    values[i] = 1.;
  }
}

#endif
