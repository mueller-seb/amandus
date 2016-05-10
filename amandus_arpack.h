/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef amandus_arpack_h
#define amandus_arpack_h

#include <amandus/amandus.h>


/**
 * A solution operator using AmandusApplicationSparse::arpack_solve().
 *
 * @ingroup apps
 */
template <int dim>
class AmandusArpack
  : public dealii::Algorithms::OperatorBase
{
  public:
    /**
     * Constructor of the operator, taking the <code>application</code>
     * and the <code>integrator</code> which is used to assemble the
     * matrices.
     */
    AmandusArpack(AmandusApplicationSparse<dim>& application,
		  AmandusIntegrator<dim>& integrator);
    /**
     * Apply the solution operator. If indicated by events, reassemble matrices 
     */
    virtual void operator() (dealii::AnyData &out, const dealii::AnyData &in);
  private:
    /// The pointer to the application object.
    dealii::SmartPointer<AmandusApplicationSparse<dim>, AmandusSolve<dim> > application;
    /// The pointer to the local integrator for assembling matrices
    dealii::SmartPointer<AmandusIntegrator<dim>, AmandusSolve<dim> > integrator;
};

#endif
