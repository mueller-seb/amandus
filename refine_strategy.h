#ifndef amandus_refine_strategy_h
#define amandus_refine_strategy_h

/**
 * @file
 *
 * @brief Refine Strategies
 *
 * @ingroup Postprocessing
 */

#include <amandus/amandus.h>
#include <amandus/integrator.h>
#include <deal.II/algorithms/operator.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/solution_transfer.h>

/**
 * @brief abstract base class AmandusRefineStrategy
 *
 * @ingroup Postprocessing
 **/
template <int dim>
class AmandusRefineStrategy
{
public:
  AmandusRefineStrategy(dealii::Triangulation<dim>& tria)
    : tria(&tria)
  {
  }
  virtual void operator()(const dealii::Vector<double>& indicator) = 0;

protected:
  dealii::SmartPointer<dealii::Triangulation<dim>, AmandusRefineStrategy<dim>> tria;
};

namespace RefineStrategy
{
/**
 * @brief mark all, i.e. uniform refinement
 *
 * @ingroup Postprocessing
 **/
template <int dim>
class MarkUniform : public AmandusRefineStrategy<dim>
{
public:
  MarkUniform(dealii::Triangulation<dim>& tria)
    : AmandusRefineStrategy<dim>(tria)
  {
  }

  void
  operator()(const dealii::Vector<double>& /*indicator*/)
  {
    this->tria->set_all_refine_flags();
  }

  void
  operator()()
  {
    this->tria->set_all_refine_flags();
  }
};

/**
 * @brief mark with maximum criterium
 *
 * @ingroup Postprocessing
 **/
template <int dim>
class MarkMaximum : public AmandusRefineStrategy<dim>
{
public:
  MarkMaximum(dealii::Triangulation<dim>& tria, double refine_threshold)
    : AmandusRefineStrategy<dim>(tria)
    , refine_threshold(refine_threshold)
  {
  }

  void
  operator()(const dealii::Vector<double>& indicator)
  {
    double threshold = this->refine_threshold * indicator.linfty_norm();
    dealii::GridRefinement::refine(*(this->tria), indicator, threshold);
  }

protected:
  double refine_threshold;
};

/**
 * @brief mark with bulk criterion
 *
 * @ingroup Postprocessing
 **/
template <int dim>
class MarkBulk : public AmandusRefineStrategy<dim>
{
public:
  MarkBulk(dealii::Triangulation<dim>& tria, double refine_threshold, double coarsen_threshold = 0.)
    : AmandusRefineStrategy<dim>(tria)
    , refine_threshold(refine_threshold)
    , coarsen_threshold(coarsen_threshold)
  {
  }

  void
  operator()(const dealii::Vector<double>& indicator)
  {
    dealii::Vector<double> square_indicators(indicator);
    square_indicators.scale(indicator);
    dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
      *(this->tria), square_indicators, this->refine_threshold, this->coarsen_threshold);
  }

protected:
  double refine_threshold, coarsen_threshold;
};

/**
 * @brief mark optimize
 *
 * @ingroup Postprocessing
 **/
template <int dim>
class MarkOptimize : public AmandusRefineStrategy<dim>
{
public:
  MarkOptimize(dealii::Triangulation<dim>& tria, const unsigned int order = 2)
    : AmandusRefineStrategy<dim>(tria)
    , order(order)
  {
  }

  void
  operator()(const dealii::Vector<double>& indicator)
  {
    dealii::GridRefinement::refine_and_coarsen_optimize(*(this->tria), indicator, this->order);
  }

protected:
  int order;
};
}

#endif
