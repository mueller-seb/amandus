#ifndef amandus_adaptivity_h
#define amandus_adaptivity_h

/**
 * @file
 *
 * @brief Utilities for adaptive mesh refinement
 *
 * @ingroup Postprocessing
 */

#include <amandus/amandus.h>
#include <amandus/integrator.h>
#include <amandus/refine_strategy.h>
#include <deal.II/algorithms/operator.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/solution_transfer.h>

/**
 * @brief Base class for remeshing operators.
 *
 * @ingroup Postprocessing
 **/
template <class VECTOR, int dim>
class Remesher : public dealii::Algorithms::OperatorBase
{
public:
  Remesher(AmandusApplicationSparse<dim>& app, dealii::Triangulation<dim>& tria)
    : app(&app)
    , tria(&tria)
  {
  }

protected:
  dealii::SmartPointer<AmandusApplicationSparse<dim>, Remesher<VECTOR, dim>> app;
  dealii::SmartPointer<dealii::Triangulation<dim>, Remesher<VECTOR, dim>> tria;
};

/**
 * @brief Remesher that interpolates stored vectors upon refinement
 *
 * @ingroup Postprocessing
 **/
template <class VECTOR, int dim>
class InterpolatingRemesher : public Remesher<VECTOR, dim>
{
public:
  InterpolatingRemesher(AmandusApplicationSparse<dim>& app, dealii::Triangulation<dim>& tria)
    : Remesher<VECTOR, dim>(app, tria)
    , transfer(app.dofs())
  {
    this->connect_transfer();
  }

  /// called at pre_refinement signal of triangulation
  virtual void
  prepare_transfer()
  {
    this->transfer.prepare_for_coarsening_and_refinement(this->to_transfer);
  }

  /**
   * @ brief called at post_refinement signal of triangulation
   *
   * Setup system on new mesh and interpolate all vectors in to_transfer
   * into transferred.
   *
   * Overload this function in a subclass if you need to notify some other
   * operator of the remeshing. (Theta timestepping reassembles anyway
   * after a timestep, thus we don't need to do it in that case)
   */
  virtual void
  finalize_transfer()
  {
    this->app->setup_system();
    unsigned int n_dofs = this->app->dofs().n_dofs();

    this->transferred.resize(this->to_transfer.size());
    for (typename std::vector<VECTOR>::iterator result = this->transferred.begin();
         result != this->transferred.end();
         ++result)
    {
      result->reinit(n_dofs);
    }
    this->transfer.interpolate(this->to_transfer, this->transferred);
    this->transfer.clear();
    for (typename std::vector<VECTOR>::iterator result = this->transferred.begin();
         result != this->transferred.end();
         ++result)
    {
      this->app->hanging_nodes().distribute(*result);
    }
  }

protected:
  void
  connect_transfer()
  {
    this->tria->signals.pre_refinement.connect(dealii::std_cxx11::bind(
      &InterpolatingRemesher<VECTOR, dim>::prepare_transfer, dealii::std_cxx11::ref(*this)));
    this->tria->signals.post_refinement.connect(dealii::std_cxx11::bind(
      &InterpolatingRemesher<VECTOR, dim>::finalize_transfer, dealii::std_cxx11::ref(*this)));
  }

  dealii::SolutionTransfer<dim, VECTOR> transfer;
  /// Vectors to be transferred to new grid.
  std::vector<VECTOR> to_transfer;
  /// Interpolated vectors on new grid.
  std::vector<VECTOR> transferred;
};

/**
 * @brief Interpolating remesher which interpolates all vectors.
 *
 * @ingroup Postprocessing
 **/
template <class VECTOR, int dim>
class AllInterpolatingRemesher : public InterpolatingRemesher<VECTOR, dim>
{
public:
  AllInterpolatingRemesher(AmandusApplicationSparse<dim>& app, dealii::Triangulation<dim>& tria)
    : InterpolatingRemesher<VECTOR, dim>(app, tria)
  {
  }

  virtual void operator()(dealii::AnyData& out, const dealii::AnyData& in)
  {
    this->extract_vectors(out);
    this->remesh(out, in);
  }

  virtual void
  extract_vectors(const dealii::AnyData& to_extract)
  {
    this->to_transfer.resize(0);
    this->extracted.resize(0);
    for (unsigned int i = 0; i < to_extract.size(); ++i)
    {
      if (to_extract.is_type<VECTOR*>(i))
      {
        this->extracted.push_back(to_extract.entry<VECTOR*>(i));
        this->to_transfer.push_back(*(to_extract.entry<VECTOR*>(i)));
      }
    }
  }

  virtual void
  finalize_transfer()
  {
    InterpolatingRemesher<VECTOR, dim>::finalize_transfer();
    for (unsigned int i = 0; i < this->extracted.size(); ++i)
    {
      dealii::deallog << "Writing back result of size " << this->transferred[i].size() << std::endl;
      *(this->extracted[i]) = this->transferred[i];
    }
  }

  virtual void remesh(const dealii::AnyData& out, const dealii::AnyData& in) = 0;

protected:
  std::vector<VECTOR*> extracted;
};

/**
 * @brief Remesher interpolating all vectors for uniform refinement.
 * Refines the mesh and interpolates all the vectors it received.
 *
 * @ingroup Postprocessing
 **/
template <class VECTOR, int dim>
class UniformRemesher : public AllInterpolatingRemesher<VECTOR, dim>
{
public:
  UniformRemesher(AmandusApplicationSparse<dim>& app, dealii::Triangulation<dim>& tria)
    : AllInterpolatingRemesher<VECTOR, dim>(app, tria)
  {
  }

  virtual void
  remesh(const dealii::AnyData& /*out*/, const dealii::AnyData& /*in*/)
  {
    this->tria->set_all_refine_flags();
    this->tria->execute_coarsening_and_refinement();
  }
};

/**
 * @brief Remesher interpolating all vectors and using an ErrorIntegrator
 * to calculate criterion.
 *
 * Calculates a cellwise refinement indicator based on an ErrorIntegrator.
 * Passes that information to a callback which should flag cells for
 * refinement/coarsening depending on the indicator. Modifies the mesh and
 * interpolates all the vectors it received.
 *
 * @ingroup Postprocessing
 **/
template <class VECTOR, int dim>
class ErrorRemesher : public AllInterpolatingRemesher<VECTOR, dim>
{
public:
  ErrorRemesher(AmandusApplicationSparse<dim>& app, dealii::Triangulation<dim>& tria,
                ErrorIntegrator<dim>& error_integrator)
    : AllInterpolatingRemesher<VECTOR, dim>(app, tria)
    , error_integrator(&error_integrator)
  {
  }

  /// Set callback for flagging
  void
  flag_callback(dealii::std_cxx11::function<void(dealii::Triangulation<dim>&,
                                                 const dealii::BlockVector<double>&)> callback)
  {
    this->callback = callback;
  }

  virtual void
  remesh(const dealii::AnyData& out, const dealii::AnyData& /*in*/)
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
  dealii::std_cxx11::function<void(dealii::Triangulation<dim>&, const dealii::BlockVector<double>&)>
    callback;
};

/**
 * @brief Remesher interpolating all vectors and using an estimator integrator
 * to calculate criterion.
 *
 * @ingroup Postprocessing
 **/
template <class VECTOR, int dim>
class EstimateRemesher : public AllInterpolatingRemesher<VECTOR, dim>
{
public:
  EstimateRemesher(AmandusApplicationSparse<dim>& app, dealii::Triangulation<dim>& tria,
                   AmandusRefineStrategy<dim>& mark, AmandusIntegrator<dim>& estimate_integrator)
    : AllInterpolatingRemesher<VECTOR, dim>(app, tria)
    , estimate_integrator(&estimate_integrator)
    , mark(mark)
  {
  }

  virtual void
  remesh(const dealii::AnyData& out, const dealii::AnyData& /*in*/)
  {
    this->app->estimate(out, *(this->estimate_integrator));
    const dealii::Vector<double> indicators = this->app->indicators();
    this->mark(indicators);
    this->tria->execute_coarsening_and_refinement();
  }

protected:
  AmandusIntegrator<dim>* estimate_integrator;
  AmandusRefineStrategy<dim>& mark;
};

#endif
