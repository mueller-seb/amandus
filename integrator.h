/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#ifndef __amandus_integrator_h
#define __amandus_integrator_h

#include <deal.II/meshworker/integration_info.h>
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

    dealii::UpdateFlags update_flags () const;
    void add_flags(const dealii::UpdateFlags flags);
 private:
    dealii::UpdateFlags u_flags;
};

namespace Integrators
{
/**
 * The local integrator for the two residuals in
 * dealii::ThetaTimestepping, namely the explicit part and the
 * Newton-residual of the implicit part.
 */
  template <int dim>
  class Theta : public AmandusIntegrator<dim>
  {
    public:
      /**
       * Constructor setting the integrator for the stationary problem
       * and optionally a BlockMask, which for DAE identifies which
       * blocks are subject to timestepping. Given a stationary
       * operator \f$ F(u) \f$, the local integral computed is
       * \f[
       * Mu \pm F(u),
       * \f]
       * where the sign is positive for the implicit operator and
       * negative for the explicit.
       *
       *
       * \param client is the integrator for the
       * stationary problem. Copied into the dealii::SmartPointer #client.
       *
       * \param implicit selects whether the Newton
       * residual of the implicit
       * side (true) or the explicit side of the theta scheme is
       * integrated. Copied into the variable #is_implicit.
       *
       * \param blocks If the system is a DAE, for
       * instance the Stokes equations, then the timestepping applies
       * only to some parts of the system, for instance only the
       * velocities. Thus, the mass matrix in the fomula above would
       * have empty blocks.
       *
       * @warning Requires that
       * dealii::MeshWorker::LoopControl::cells_first is false! This
       * integrator has to modify the results of the stationary
       * integrator. In part, for face integration, this modification
       * can be applied in face(). But since the matrix coupling only
       * interior degrees of freedom is accumulated through all
       * interior and boundary faces, the modification of scaling the
       * stationary matrix and then adding the mass matrix <b>must</b>
       * be applied after all these contributions have been
       * assembled. Setting
       * dealii::MeshWorker::LoopControl::cells_first to false
       * accompliches this.
       */
      Theta (AmandusIntegrator<dim>& client,
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
      dealii::SmartPointer<AmandusIntegrator<dim>,  Theta<dim> > client;
      bool is_implicit;
      dealii::BlockMask block_mask;
  };  
}

// TODO: Update flags should be set by derived classes with add_flags. Right
// now some update flags are always set by this constructor. For
// compatibility keep it this way until the applications are adjusted.
template <int dim>
inline
AmandusIntegrator<dim>::AmandusIntegrator ()
:
timestep(0.)  , u_flags(dealii::update_JxW_values |
			dealii::update_values |
			dealii::update_gradients |
			dealii::update_quadrature_points)
{}


template <int dim>
inline
dealii::UpdateFlags
AmandusIntegrator<dim>::update_flags () const
{
  return u_flags;
}


template <int dim>
inline
void
AmandusIntegrator<dim>::add_flags (const dealii::UpdateFlags flags)
{
  u_flags |= flags;
}


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
  Theta<dim>::Theta(AmandusIntegrator<dim>& cl,
		    bool implicit,
		    dealii::BlockMask blocks)
: client(&cl), block_mask(blocks)
{
  is_implicit = implicit;
  this->use_cell = client->use_cell;
  this->use_boundary = client->use_boundary;
  this->use_face = client->use_face;
  // Copy vector requests
}


  template <int dim>
  inline
  void
  Theta<dim>::extract_data (const dealii::AnyData& data)
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
  Theta<dim>::cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
		   dealii::MeshWorker::IntegrationInfo<dim>& info) const
  {
    client->cell(dinfo, info);
    const double factor = is_implicit ? this->timestep : -this->timestep;
    for (unsigned int i=0;i<dinfo.n_vectors();++i)
      dinfo.vector(i) *= factor;
    for (unsigned int i=0;i<dinfo.n_matrices();++i)
      dinfo.matrix(i, false).matrix *= factor;
    
    const dealii::FiniteElement<dim>& fe = info.finite_element();
    
    unsigned int comp = 0;
    for (unsigned int b=0;b<fe.n_base_elements();++b)
      {
	unsigned int k=fe.first_block_of_base(b);
	const dealii::FiniteElement<dim>& base = fe.base_element(b);
	for (unsigned int m=0;m<fe.element_multiplicity(b);++m)
	  {
	    for (unsigned int i=0;i<dinfo.n_vectors();++i)
	      dealii::LocalIntegrators::L2::L2(
		dinfo.vector(i).block(k+m), info.fe_values(b),
		dealii::make_slice(info.values[0], comp, base.n_components()));
	    
	    for (unsigned int i=0;i<dinfo.n_matrices();++i)
	      if (dinfo.matrix(i, false).row == k+m
		  && dinfo.matrix(i, false).column == k+m)
		dealii::LocalIntegrators::L2::mass_matrix(
		  dinfo.matrix(i, false).matrix, info.fe_values(b));
	    comp += base.n_components();
	  }
      }
  }
  
  template <int dim>
  void
  Theta<dim>::boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
		       dealii::MeshWorker::IntegrationInfo<dim>& info) const
  {
    client->boundary(dinfo, info);
    const double factor = is_implicit ? this->timestep : -this->timestep;
    for (unsigned int i=0;i<dinfo.n_vectors();++i)
      dinfo.vector(i) *= factor;
    for (unsigned int i=0;i<dinfo.n_matrices();++i)
      dinfo.matrix(i, false).matrix *= factor;
  }
  
  template <int dim>
  void
  Theta<dim>::face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
		   dealii::MeshWorker::DoFInfo<dim>& dinfo2,
		   dealii::MeshWorker::IntegrationInfo<dim>& info1,
		   dealii::MeshWorker::IntegrationInfo<dim>& info2) const
  {
    client->face(dinfo1, dinfo2, info1, info2);
    const double factor = is_implicit ? this->timestep : -this->timestep;
    for (unsigned int i=0;i<dinfo2.n_vectors();++i)
      {
	dinfo1.vector(i) *= factor;
	dinfo2.vector(i) *= factor;
      }
    for (unsigned int i=0;i<dinfo1.n_matrices();++i)
      {
	dinfo1.matrix(i, false).matrix *= factor;
	dinfo1.matrix(i, true).matrix *= factor;
	dinfo2.matrix(i, false).matrix *= factor;
	dinfo2.matrix(i, true).matrix *= factor;
      }
  }
}



#endif











