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
#include <deal.II/integrators/l2.h>
#include <deal.II/base/vector_slice.h>

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

namespace Integrators
{
/**
 *
 */
  template <int dim>
  class ThetaResidual : public AmandusIntegrator<dim>
  {
    public:
      /**
       * Constructor setting the integrator for the stationary problem
       * and optionally a BlockMask, which for DAE identifies which
       * blocks are subject to timestepping.
       */
      ThetaResidual (AmandusIntegrator<dim>& client,
		     bool implicit,
		     dealii::BlockMask blocks = dealii::BlockMask());
     virtual void extract_data (const dealii::AnyData& data);
   private:
      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
			dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
			    dealii::MeshWorker::IntegrationInfo<dim>& info) const;
      virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
			dealii::MeshWorker::DoFInfo<dim>& dinfo2,
			dealii::MeshWorker::IntegrationInfo<dim>& info1,
			dealii::MeshWorker::IntegrationInfo<dim>& info2) const;
      dealii::SmartPointer<AmandusIntegrator<dim>,  ThetaResidual<dim> > client;
      bool is_implicit;
      dealii::BlockMask block_mask;
  };
}


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


namespace Integrators
{
  template <int dim>
  inline
  ThetaResidual<dim>::ThetaResidual(AmandusIntegrator<dim>& client,
				    bool implicit,
				    dealii::BlockMask blocks)
: client(&client), block_mask(blocks)
{
  is_implicit = implicit;
}


  template <int dim>
  inline
  void
  ThetaResidual<dim>::extract_data (const dealii::AnyData& data)
  {
    client->extract_data(data);
    const double* ts = data.try_read_ptr<double>("Timestep");
    if (ts != 0)
      {
	this->timestep = *ts;
      }
  }
  
  
  template <int dim>
  void
  ThetaResidual<dim>::cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
			   dealii::MeshWorker::IntegrationInfo<dim>& info) const
  {
    client->cell(dinfo, info);
    const double factor = is_implicit ? this->timestep : -this->timestep;
    dinfo.vector(0) *= factor;

    // Todo: this is wrong, since fe_values should only have the index
    // of the base element for block b.
    const dealii::FiniteElement<dim>& fe = info.finite_element();

    unsigned int comp = 0;
    for (unsigned int b=0;b<fe.n_base_elements();++b)
      {
	unsigned int k=fe.first_block_of_base(b);
	const dealii::FiniteElement<dim>& base = fe.base_element(b);
	for (unsigned int m=0;m<fe.element_multiplicity(b);++m)
	  {
	    dealii::LocalIntegrators::L2::L2(
	      dinfo.vector(0).block(k+m), info.fe_values(b),
	      dealii::make_slice(info.values[0], comp, base.n_components()));
	    comp += base.n_components();
	  }
      }
  }
  
  template <int dim>
  void
  ThetaResidual<dim>::boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
			       dealii::MeshWorker::IntegrationInfo<dim>& info) const
  {
    client->boundary(dinfo, info);
    const double factor = is_implicit ? this->timestep : -this->timestep;
    dinfo.vector(0) *= factor;
}
  
  template <int dim>
  void
  ThetaResidual<dim>::face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
			   dealii::MeshWorker::DoFInfo<dim>& dinfo2,
			   dealii::MeshWorker::IntegrationInfo<dim>& info1,
			   dealii::MeshWorker::IntegrationInfo<dim>& info2) const
  {
    client->face(dinfo1, dinfo2, info1, info2);
    const double factor = is_implicit ? this->timestep : -this->timestep;
    dinfo1.vector(0) *= factor;
    dinfo2.vector(0) *= factor;
  }
  

}



#endif

