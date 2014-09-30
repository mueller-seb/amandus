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
 * Computes the boundary values,  
 * with two different boundary parts.
 * 
 * 
 */

template <int dim>
class BoundaryValues: public Function<dim>
{
	public:
	BoundaryValues () {};
	virtual void value_list (const std::vector<Point<dim> > &points,
				std::vector<double> &values,
				const unsigned int component=0) const;
	};
template <int dim>
void BoundaryValues<dim>::value_list(const std::vector<Point<dim> > &points,
					std::vector<double> &values,
					const unsigned int) const
{
	Assert(values.size()==points.size(),
	ExcDimensionMismatch(values.size(),points.size()));
	for (unsigned int i=0; i<values.size(); ++i)
	{	
		if (points[i](0)<0.5)
		values[i]=0.;
		else
		values[i]=1.;
	}
}


#endif
