/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __amandus_integrator_h
#define __amandus_integrator_h

#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/fe/block_mask.h>
#include <deal.II/algorithms/any_data.h>
#include <deal.II/base/smartpointer.h>

template <int dim>
class AmandusIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
{
  public:
    /**
     * Empty constructor
     */
    AmandusIntegrator ();
    /**
     * Extract data which does not depend on the current cell from
     * AnyData, before the loop over cells and faces is run.
     */
    virtual void extract_data (const dealii::AnyData& data);
    double timestep;
};

/**
 *
 */
template <int dim>
class ThetaImplicitIntegrator : public AmandusIntegrator<dim>
{
  public:
    /**
     * Constructor setting the integrator for the stationary problem
     * and optionally a BlockMask, which for DAE identifies which
     * blocks are subject to timestepping.
     */
    ThetaImplicitIntegrator (AmandusIntegrator<dim>& client, dealii::BlockMask blocks = dealii::BlockMask());
  private:
    virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
		      dealii::MeshWorker::IntegrationInfo<dim>& info) const;
    virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
			  dealii::MeshWorker::IntegrationInfo<dim>& info) const;
    virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
		      dealii::MeshWorker::DoFInfo<dim>& dinfo2,
		      dealii::MeshWorker::IntegrationInfo<dim>& info1,
		      dealii::MeshWorker::IntegrationInfo<dim>& info2) const;
    dealii::SmartPointer<AmandusIntegrator<dim>,  ThetaImplicitIntegrator<dim> > client;
    dealii::BlockMask block_mask;
};


template <int dim>
inline
AmandusIntegrator<dim>::AmandusIntegrator ()
		:
		timestep(0.)
{}


template <int dim>
inline
void
AmandusIntegrator<dim>::extract_data (const dealii::AnyData& data)
{
  const double* ts = data.try_read_ptr<double>("Timestep");
  if (ts != 0)
    {
      timestep = *ts;
    }
}


template <int dim>
inline
ThetaImplicitIntegrator<dim>::ThetaImplicitIntegrator(AmandusIntegrator<dim>& client, dealii::BlockMask blocks)
: client(&client), block_mask(blocks)
{}



#endif

