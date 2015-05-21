#ifndef __adaptivity_h
#define __adaptivity_h

#include <amandus.h>
#include <integrator.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/algorithms/operator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/grid/grid_refinement.h>

/**
 * Base class for remeshing operators. Has access to app's triangulation.
 **/
template <class VECTOR, int dim>
class Remesher : public dealii::Algorithms::OperatorBase
{
  public:
    Remesher(AmandusApplicationSparse<dim>& app,
             dealii::Triangulation<dim>& tria) : app(&app), tria(&tria)
  {}

  protected:
    dealii::SmartPointer<
      AmandusApplicationSparse<dim>,
      Remesher<VECTOR, dim> > app;
    dealii::SmartPointer<
      dealii::Triangulation<dim>,
      Remesher<VECTOR, dim> > tria;
};

/**
 * Remesher which is listening for refinements and interpolates given
 * vectors.
 **/
template <class VECTOR, int dim>
class InterpolatingRemesher : public Remesher<VECTOR, dim>
{
  public:
    InterpolatingRemesher(AmandusApplicationSparse<dim>& app,
                          dealii::Triangulation<dim>& tria) :
      Remesher<VECTOR, dim>(app, tria),
      transfer(app.dofs())
  {
    this->connect_transfer();
  }

    void connect_transfer()
    {
      this->tria->signals.pre_refinement.connect(
          dealii::std_cxx11::bind(
              &InterpolatingRemesher<VECTOR, dim>::prepare_transfer,
              dealii::std_cxx11::ref(*this)));
      this->tria->signals.post_refinement.connect(
          dealii::std_cxx11::bind(
              &InterpolatingRemesher<VECTOR, dim>::finalize_transfer,
              dealii::std_cxx11::ref(*this)));
    }

    virtual void prepare_transfer()
    {
      this->transfer.prepare_for_coarsening_and_refinement(this->to_transfer);
    }

    virtual void finalize_transfer()
    {
      this->app->setup_system();
      unsigned int n_dofs = this->app->dofs().n_dofs();

      this->transferred.resize(this->to_transfer.size());
      for(typename std::vector<VECTOR>::iterator result = this->transferred.begin();
          result != this->transferred.end();
          ++result)
      {
        result->reinit(n_dofs);
      }
      this->transfer.interpolate(this->to_transfer, this->transferred);
      this->transfer.clear();
    }

  protected:
    dealii::SolutionTransfer<dim, VECTOR> transfer;
    std::vector<VECTOR> to_transfer;
    std::vector<VECTOR> transferred;
};


/**
 * Remesher which interpolates all vectors from out.
 **/
template <class VECTOR, int dim>
class AllInterpolatingRemesher : public InterpolatingRemesher<VECTOR, dim>
{
  public:
    AllInterpolatingRemesher(AmandusApplicationSparse<dim>& app,
                             dealii::Triangulation<dim>& tria) :
      InterpolatingRemesher<VECTOR, dim>(app, tria)
  {}

    virtual void operator()(dealii::AnyData& out,
                            const dealii::AnyData& in)
    {
      this->extract_vectors(out);
      this->remesh(out, in);
    }

    virtual void extract_vectors(const dealii::AnyData& to_extract)
    {
      this->to_transfer.resize(0);
      this->extracted.resize(0);
      for(unsigned int i = 0; i < to_extract.size(); ++i)
      {
        if(to_extract.is_type<VECTOR*>(i))
        {
          this->extracted.push_back(to_extract.entry<VECTOR*>(i));
          this->to_transfer.push_back(*(to_extract.entry<VECTOR*>(i)));
        }
      }
    }

    virtual void finalize_transfer()
    {
      InterpolatingRemesher<VECTOR, dim>::finalize_transfer();
      for(unsigned int i = 0; i < this->extracted.size(); ++i)
      {
        dealii::deallog << "Writing back result of size " << this->transferred[i].size() << std::endl;
        *(this->extracted[i]) = this->transferred[i];
      }
    }

    virtual void remesh(const dealii::AnyData& out,
                        const dealii::AnyData& in) = 0;

  protected:
    std::vector<VECTOR*> extracted;
};


/**
 * Remesher which uses an Error Integrator to calculate refinement criterion
 **/
template <class VECTOR, int dim>
class ErrorRemesher : public AllInterpolatingRemesher<VECTOR, dim>
{
  public:
    ErrorRemesher(AmandusApplicationSparse<dim>& app,
                  dealii::Triangulation<dim>& tria,
                  ErrorIntegrator<dim>& error_integrator) :
      AllInterpolatingRemesher<VECTOR, dim>(app, tria),
      error_integrator(&error_integrator)
  {}

    void flag_callback(
        dealii::std_cxx11::function<
        void(dealii::Triangulation<dim>&, const dealii::BlockVector<double>&)>
        callback)
    {
      this->callback = callback;
    }

    virtual void remesh(const dealii::AnyData& out,
                        const dealii::AnyData& in)
    {
      this->app->error(indicator, out, *(this->error_integrator));
      this->callback(*(this->tria), indicator);
      this->tria->execute_coarsening_and_refinement();
    }

  protected:
    /*
    dealii::SmartPointer<
      ErrorIntegrator<dim>,
      ErrorRemesher<VECTOR, dim> > error_integrator; //TODO SmartPointer
      */
    ErrorIntegrator<dim>* error_integrator;
    dealii::BlockVector<double> indicator;
    dealii::std_cxx11::function<
      void(dealii::Triangulation<dim>&,
           const dealii::BlockVector<double>&)> callback;
};

#endif
