/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __amandus_h
#define __amandus_h

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/relaxation_block.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_list.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/multigrid/mg_dof_handler.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/output.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>

#include <deal.II/base/flow_function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/named_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/algorithms/operator.h>

#include <iostream>
#include <fstream>

/**
 * An application class with solvers based on local Schwarz smoothers
 * without strong boundary conditions.
 *
 * This class provides storage for the DoFHandler object and the
 * matrices associated with a finite element discretization and its
 * multilevel solver. The basic structures only depend on the
 * Triangulation and the FiniteElement, which are provided to the
 * constructor.
 *
 * @todo: Straighten up the interface, make private things private
 *
 * @todo: Create interface for ParameterHandler to set parameters for
 * #control, possibly select solver and other parameters
 *
 * @author Guido Kanschat
 * @date 2014
 */
template <int dim>
class AmandusApplicationBase : public dealii::Subscriptor  
{
public:
  typedef dealii::MeshWorker::IntegrationInfo<dim> CellInfo;
  
  /**
   * Constructor, setting the finite element and the
   * triangulation. This constructor does not distribute the degrees
   * of freedom or initialize sparsity patterns, which has to be
   * achieved by calling setup_system().
   */
  AmandusApplicationBase(dealii::Triangulation<dim>& triangulation,
			 const dealii::FiniteElement<dim>& fe);

  /**
   * Initialize the vector <code>v</code> to the size matching the
   * DoFHandler. This requires that setup_system() is called before.
   */
  void setup_vector (dealii::Vector<double>& v) const;
  
  /**
   * Initialize the finite element system on the current mesh.  This
   * involves distributing degrees of freedom for the leaf mesh and
   * the levels as well as setting up sparsity patterns for the matrices.
   *
   * @note This function calls the virtual function setup_constraints().
   */
  void setup_system ();
  
  /**
   * Apply the local operator <code>integrator</code> to the vectors
   * in <code>in</code> and write the result to the first vector in
   * <code>out</code>.
   */
  void assemble_right_hand_side(const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
				dealii::NamedData<dealii::Vector<double> *> &out,
				const dealii::NamedData<dealii::Vector<double> *> &in) const;
  
  void refine_mesh (const bool global = false);
  
public:
  /**
   * Set up hanging node constraints for leaf mesh and for level
   * meshes. Use redefinition in derived classes to add boundary
   * constraints.
   */
  virtual void setup_constraints ();
  /**
   * Use the integrator to build the matrix for the leaf mesh.
   */
  void assemble_matrix (const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
			const dealii::NamedData<dealii::Vector<double> *> &in);
  /**
   * Use the integrator to build the matrix for the level meshes. This
   * also automatically generates the transfer matrices needed for
   * multigrid with local smoothing on locally refined meshes.
   */
    void assemble_mg_matrix (const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
			     const dealii::NamedData<dealii::Vector<double> *> &in);
  /**
   * Currently disabled.
   */
  double estimate(const dealii::MeshWorker::LocalIntegrator<dim>& integrator);
  /**
   * Compute several error values using the integrator. The number
   * of errors computed is given as the last argument.
   *
   * @todo Improve the interface to determine the number of errors from the integrator.
   */
  void error (const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
	      const dealii::NamedData<dealii::Vector<double> *> &in,
	      unsigned int num_errs);
  /**
   * Solve the linear system stored in #matrix with the right hand
   * side given. Uses the multigrid preconditioner.
   */
  void solve (dealii::Vector<double>& sol, const dealii::Vector<double>& rhs);
  void output_results(unsigned int refinement_cycle,
		      const dealii::NamedData<dealii::Vector<double>*>* data = 0) const;
  
  /**
   * For testing purposes compare the residual operator applied to to
   * the vectors in <code>in</code> with the result of the matrix
   * vector multiplication.
   */
  void verify_residual(const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
		       dealii::NamedData<dealii::Vector<double> *> &out,
		       const dealii::NamedData<dealii::Vector<double> *> &in) const;
  
  /**
   * The object controlling the iteration in solve().
   */
  dealii::ReductionControl control;
  //  protected:
  dealii::SmartPointer<dealii::Triangulation<dim>, AmandusApplicationBase<dim> > triangulation;
  const dealii::MappingQ1<dim>      mapping;
  dealii::SmartPointer<const dealii::FiniteElement<dim>, AmandusApplicationBase<dim> > fe;
  dealii::MGDoFHandler<dim>         mg_dof_handler;
  dealii::DoFHandler<dim>&          dof_handler;
  
  dealii::ConstraintMatrix     constraints;
  dealii::MGConstrainedDoFs    mg_constraints;
  
  dealii::SparsityPattern      sparsity;
  dealii::SparseMatrix<double> matrix;
  dealii::Vector<double>       solution;
  dealii::Vector<double>       right_hand_side;
  dealii::BlockVector<double>  estimates;
  dealii::MGLevelObject<dealii::SparsityPattern> mg_sparsity;
  dealii::MGLevelObject<dealii::SparseMatrix<double> > mg_matrix;
  
  dealii::MGLevelObject<dealii::SparseMatrix<double> > mg_matrix_down;
  dealii::MGLevelObject<dealii::SparseMatrix<double> > mg_matrix_up;
  
  dealii::MGTransferPrebuilt<dealii::Vector<double> > mg_transfer;
};


/**
 * The same as AmandusApplicationBase, but with multigrid constraints
 * and homogeneous Dirichlet boundary conditions.
 */
template <int dim>
class AmandusApplication : public AmandusApplicationBase<dim>
{
 public:
  AmandusApplication(dealii::Triangulation<dim>& triangulation,
		     const dealii::FiniteElement<dim>& fe);
 private:
  virtual void setup_constraints ();
};


/**
 * A residual operator using AmandusApplicationBase::assemble_right_hand_side().
 */
template <int dim>
class AmandusResidual
  : public dealii::Algorithms::Operator<dealii::Vector<double> >
{
  public:
    AmandusResidual(const AmandusApplicationBase<dim>& application,
		    const dealii::MeshWorker::LocalIntegrator<dim>& integrator);
		    
    virtual void operator() (dealii::NamedData<dealii::Vector<double> *> &out,
			     const dealii::NamedData<dealii::Vector<double> *> &in);
  private:
    dealii::SmartPointer<const AmandusApplicationBase<dim>, AmandusResidual<dim> > application;
    dealii::SmartPointer<const dealii::MeshWorker::LocalIntegrator<dim>, AmandusResidual<dim> > integrator;
};

/**
 * A solution operator using AmandusApplicationBase::solve().
 */
template <int dim>
class AmandusSolve
  : public dealii::Algorithms::Operator<dealii::Vector<double> >
{
public:
  /**
   * Constructor of the operator, taking the <code>application</code>
   * and the <code>integrator</code> which is used to assemble the
   * matrices.
   */
  AmandusSolve(AmandusApplicationBase<dim>& application,
	       const dealii::MeshWorker::LocalIntegrator<dim>& integrator);
  /**
   * Apply the solution operator. If indecated by events, reassemble matrices 
   */
  virtual void operator() (dealii::NamedData<dealii::Vector<double> *> &out,
			   const dealii::NamedData<dealii::Vector<double> *> &in);
private:
  /// The pointer to the application object.
  dealii::SmartPointer<AmandusApplicationBase<dim>, AmandusSolve<dim> > application;
  /// The pointer to the local integrator for assembling matrices
  dealii::SmartPointer<const dealii::MeshWorker::LocalIntegrator<dim>, AmandusSolve<dim> > integrator;
};


#endif
