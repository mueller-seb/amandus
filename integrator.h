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
#include <deal.II/base/function.h>

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

/**
 * A container for different error integrators. The elements of this
 * container must be instances of specializations of this class.
 *
 * Derived error integrators should calculate an error value between the
 * exact solution provided by the solution member for the components
 * indicated by the component_mask and store this value in the block_idx'th
 * dinfo.value.
 **/
template <int dim>
class ErrorIntegrator : public AmandusIntegrator<dim>
{
  public:
    ErrorIntegrator()
    {}

    ErrorIntegrator(const dealii::Function<dim>& solution) : solution(&solution)
    {
      this->use_cell = false;
      this->use_face = false;
      this->use_boundary = false;
    }

    unsigned int size() const
    {
      return error_integrators.size();
    }

    void add(ErrorIntegrator<dim>* error_integrator)
    {
      dealii::ComponentMask component_mask(this->solution->n_components, true);
      this->add(error_integrator, component_mask);
    }

    void add(ErrorIntegrator<dim>* error_integrator,
             dealii::ComponentMask component_mask)
    {
      error_integrator->component_mask = component_mask;
      error_integrator->solution = this->solution;
      error_integrator->block_idx = error_integrators.size();
      error_integrators.push_back(error_integrator);
      this->add_flags(error_integrator->update_flags());
      this->use_cell = this->use_cell || error_integrator->use_cell;
      this->use_face = this->use_face || error_integrator->use_face;
      this->use_boundary = this->use_boundary || error_integrator->use_boundary;
    }

    virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                      dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {
      for(std::size_t i = 0; i < error_integrators.size(); ++i)
      {
        if(error_integrators[i]->use_cell)
        {
          error_integrators[i]->cell(dinfo, info);
        }
      }
    }

    virtual void boundary(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                          dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {
      for(std::size_t i = 0; i < error_integrators.size(); ++i)
      {
        if(error_integrators[i]->use_boundary)
        {
          error_integrators[i]->boundary(dinfo, info);
        }
      }
    }

    virtual void face(dealii::MeshWorker::DoFInfo<dim>& dinfo1,
                      dealii::MeshWorker::DoFInfo<dim>& dinfo2,
                      dealii::MeshWorker::IntegrationInfo<dim>& info1,
                      dealii::MeshWorker::IntegrationInfo<dim>& info2) const
    {
      for(std::size_t i = 0; i < error_integrators.size(); ++i)
      {
        if(error_integrators[i]->use_face)
        {
          error_integrators[i]->face(dinfo1, dinfo2,
                                     info1, info2);
        }
      }
    }

  protected:
    dealii::ComponentMask component_mask;
    unsigned int block_idx;
    const dealii::Function<dim>* solution;

  private:
    std::vector<ErrorIntegrator<dim>* > error_integrators;
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
       * operator \f$ F(u) \f$ and a weighted timestep \f$ \omega \f$,
       * the local integral computed is
       * \f[
       * Mu \pm \omega F(u),
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
       * \param enforce_homogenity Set all components corresponding to
       * blocks which are not timestepped to zero. If the initial system
       * satisfies the algebraic constraints then this will be true anyway
       * but setting it allows us to choose arbitrary initial values for
       * non-timestepped blocks.
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
       * accomplishes this.
       */
      Theta (AmandusIntegrator<dim>& client,
	     bool implicit,
	     dealii::BlockMask blocks = dealii::BlockMask(),
       bool enforce_homogenity = false);
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
      bool enforce_homogenity;
  };


  template <int dim>
    class L2ErrorIntegrator : public ErrorIntegrator<dim>
  {
    public:
      L2ErrorIntegrator()
      {
        this->use_cell = true;
        this->use_face = false;
        this->use_boundary = false;
        this->add_flags(dealii::update_JxW_values |
                        dealii::update_values |
                        dealii::update_quadrature_points);
      }
      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;
  };

  template <int dim>
    void L2ErrorIntegrator<dim>::cell(
        dealii::MeshWorker::DoFInfo<dim>& dinfo, 
        dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {
      Assert(info.values.size() >= 1,
             dealii::ExcDimensionMismatch(info.values.size(), 1));

      const unsigned int fev_idx = 0;
      const unsigned int approximation_idx = 0;

      const std::vector<dealii::Point<dim> >& q_points =
        info.fe_values(fev_idx).get_quadrature_points();
      const std::vector<double>& q_weights =
        info.fe_values(fev_idx).get_JxW_values();

      std::vector<double> solution_values(q_points.size());
      for(unsigned int component = 0;
          component < this->component_mask.size();
          ++component)
      {
        if(this->component_mask[component])
        {
          this->solution->value_list(q_points, solution_values, component);
          const std::vector<double>& approximation_values =
            info.values[approximation_idx][component];
          for(std::size_t q = 0; q < q_points.size(); ++q)
          {
            double error = solution_values[q] - approximation_values[q];
            dinfo.value(this->block_idx) += error*error * q_weights[q];
          }
        }
      }
      dinfo.value(this->block_idx) = std::sqrt(dinfo.value(this->block_idx));
    }

  template <int dim>
    class H1ErrorIntegrator : public ErrorIntegrator<dim>
  {
    public:
      H1ErrorIntegrator()
      {
        this->use_cell = true;
        this->use_face = false;
        this->use_boundary = false;
        this->add_flags(dealii::update_JxW_values |
                        dealii::update_gradients |
                        dealii::update_quadrature_points);
      }

      virtual void cell(dealii::MeshWorker::DoFInfo<dim>& dinfo,
                        dealii::MeshWorker::IntegrationInfo<dim>& info) const;
  };

  template <int dim>
    void H1ErrorIntegrator<dim>::cell(
        dealii::MeshWorker::DoFInfo<dim>& dinfo, 
        dealii::MeshWorker::IntegrationInfo<dim>& info) const
    {
      Assert(info.values.size() >= 1,
             dealii::ExcDimensionMismatch(info.values.size(), 1));

      const unsigned int fev_idx = 0;
      const unsigned int approximation_idx = 0;

      const std::vector<dealii::Point<dim> >& q_points =
        info.fe_values(fev_idx).get_quadrature_points();
      const std::vector<double>& q_weights =
        info.fe_values(fev_idx).get_JxW_values();

      std::vector<dealii::Tensor<1, dim, double> > solution_grads(q_points.size());
      for(unsigned int component = 0;
          component < this->component_mask.size();
          ++component)
      {
        if(this->component_mask[component])
        {
          this->solution->gradient_list(q_points, solution_grads, component);
          const std::vector<dealii::Tensor<1, dim, double> >& approximation_grads =
            info.gradients[approximation_idx][component];
          for(std::size_t q = 0; q < q_points.size(); ++q)
          {
            double error = (solution_grads[q] - approximation_grads[q]).norm();
            dinfo.value(this->block_idx) += error*error * q_weights[q];
          }
        }
      }
      dinfo.value(this->block_idx) = std::sqrt(dinfo.value(this->block_idx));
    }
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
		    dealii::BlockMask blocks,
        bool enforce_homogenity)
: client(&cl), block_mask(blocks), enforce_homogenity(enforce_homogenity)
{
  is_implicit = implicit;
  this->use_cell = client->use_cell;
  this->use_boundary = client->use_boundary;
  this->use_face = client->use_face;

  this->add_flags(client->update_flags());
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
        unsigned int block_idx = k + m;
        if(this->block_mask[block_idx]) // add mass operator
        {
          for (unsigned int i=0;i<dinfo.n_vectors();++i)
          {
            dealii::LocalIntegrators::L2::L2(
                dinfo.vector(i).block(block_idx), info.fe_values(b),
                dealii::make_slice(info.values[0], comp, base.n_components()));
          }

          for (unsigned int i=0;i<dinfo.n_matrices();++i)
          {
            if (dinfo.matrix(i, false).row == block_idx
                && dinfo.matrix(i, false).column == block_idx)
            {
              dealii::LocalIntegrators::L2::mass_matrix(
                  dinfo.matrix(i, false).matrix, info.fe_values(b));
            }
          }
        } else if(this->enforce_homogenity && !this->is_implicit)
        {
          for (unsigned int i=0;i<dinfo.n_vectors();++i)
          {
            dinfo.vector(i).block(block_idx) = 0.0;
          }
        }
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











