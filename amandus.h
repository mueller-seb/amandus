/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#ifndef __amandus_h
#define __amandus_h

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
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

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/flow_function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/algorithms/operator.h>

#include <integrator.h>

#include <iostream>
#include <fstream>

/**
 * A class managing a common ParameterHandler for most applications.
 */
class AmandusParameters : public dealii::ParameterHandler
{
 public:
  /**
   * Constructor declaring the default parameters. After this, but before calling read(),
   * additional parameters can be declared using the functions of the base class.
   */
  AmandusParameters ();
  
  /**
   * Read parameters from default file, if it exists, from a file
   * derived from the name of the executable by adding the suffix
   * ".prm", or from a file specified on the command line.
   */
  void read(int argc, const char** argv);
};

/**
 * An application class with a plain GMRES solver and optional UMFPack
 * as a preconditioner (see constructor arguments). This is mostly to
 * test discretizations and it sould be improved by a derived class
 * with a multigrid solver like AmandusApplication.
 *
 * This class provides storage for the DoFHandler object and the
 * matrix associated with a finite element discretization. The basic
 * structures only depend on the Triangulation and the FiniteElement,
 * which are provided to the constructor.
 *
 * The purpose of this class is not so much implementing an
 * application program, but providing the data structures used by all
 * application programs with the following characteristics:
 * <ol>
 * <li>Single vector and single matrix, no block structures; this does not exclude systems.</li>
 * <li>A single sparse matrix and accordingly single sparse matrices for for each level.</li>
 * </ol>
 *
 * @todo: Straighten up the interface, make private things private,
 * protected things protected
 *
 * @todo: Create interface for ParameterHandler to set parameters for
 * #control, possibly select solver and other parameters
 *
 * @author Guido Kanschat
 * @date 2014
 *
 * @ingroup apps
 */
template <int dim>
class AmandusApplicationSparse : public dealii::Subscriptor  
{
  public:
    typedef dealii::MeshWorker::IntegrationInfo<dim> CellInfo;
  
    /**
     * Constructor, setting the finite element and the
     * triangulation. This constructor does not distribute the degrees
    * of freedom or initialize sparsity patterns, which has to be
    * achieved by calling setup_system().
    *
    * If the argument <tt>use_umfpack</tt> is true, assemble_matrix() not only generates a matrix,
    * but also the inverse, using SparseDirectUMFPACK.
    */
    AmandusApplicationSparse(dealii::Triangulation<dim>& triangulation,
			     const dealii::FiniteElement<dim>& fe,
			     bool use_umfpack = false);

    /**
     * Parse the paramaters from a handler
     */
    void parse_parameters(dealii::ParameterHandler& param);

    /**
     * Set the boundary components that should be constrained if the
     * boundary indicator is equal to index.
     *
     * The constraints for these boundary values are set in
     * setup_constraints(), and they are always constrained to
     * zero. Inhomogeneous boundary conditions are obtained by setting
     * the boundary values of the start vector of a dealii::Newton or a
     * dealii::ThetaTimestepping method.
     *
     * @param `index`: the boundary indicator for which the boundary
     * constraints re being set.
     *
     * @param `mask`: the object selecting the blocks of an dealii::FESystem
     * to which the constraints are to be applied.
     */
    void set_boundary (unsigned int index, dealii::ComponentMask mask = dealii::ComponentMask());
    
    /**
     * Initialize the vector <code>v</code> to the size matching the
     * DoFHandler. This requires that setup_system() is called before.
     */
    virtual void setup_vector (dealii::Vector<double>& v) const;
  
    /**
     * Initialize the finite element system on the current mesh.  This
     * involves distributing degrees of freedom for the leaf mesh and
     * the levels as well as setting up sparsity patterns for the matrices.
     *
     * @note This function calls the virtual function setup_constraints().
     */
    virtual void setup_system ();
  
    /**
     * Apply the local operator <code>integrator</code> to the vectors
     * in <code>in</code> and write the result to the first vector in
     * <code>out</code>.
     */
    void assemble_right_hand_side(dealii::AnyData &out,
				  const dealii::AnyData &in,
				  const AmandusIntegrator<dim>& integrator) const;
  
    void refine_mesh (const bool global = false);

    /**
     * The object describing the finite element space.
     */
    const dealii::DoFHandler<dim>& dofs () const;
    
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
    void assemble_matrix (const dealii::AnyData &in,
			  const AmandusIntegrator<dim>& integrator);
    /**
     * Empty function here, but it is reimplemented in AmandusApplicationMultigrid.
     */
    virtual void assemble_mg_matrix (const dealii::AnyData &in,
				     const AmandusIntegrator<dim>& integrator);

    /**
     * Currently disabled.
     *
     * \todo: Make sure it takes an AnyData with a vector called "solution".
     */
    double estimate(const dealii::AnyData &in,
		    const AmandusIntegrator<dim>& integrator);
    /**
     * Compute several error values using the integrator. The number
     * of errors computed is given as the last argument.
     *
     * @todo Improve the interface to determine the number of errors from the integrator.
     */
    void error (const dealii::AnyData &in,
		const AmandusIntegrator<dim>& integrator,
		unsigned int num_errs);
    /**
     * Solve the linear system stored in #matrix with the right hand
     * side given. Uses the multigrid preconditioner.
     */
    virtual void solve (dealii::Vector<double>& sol, const dealii::Vector<double>& rhs);
    
    void output_results(unsigned int refinement_cycle,
			const dealii::AnyData* data = 0) const;
  
    /**
     * For testing purposes compare the residual operator applied to to
     * the vectors in <code>in</code> with the result of the matrix
     * vector multiplication.
     */
    void verify_residual(dealii::AnyData &out,
			 const dealii::AnyData &in,
			 const AmandusIntegrator<dim>& integrator) const;
  
    /**
     * The object controlling the iteration in solve().
     */
    dealii::ReductionControl control;
    
  protected:
    /// The mesh
    dealii::SmartPointer<dealii::Triangulation<dim>, AmandusApplicationSparse<dim> > triangulation;
    
    /// The default mapping
    const dealii::MappingQ1<dim>      mapping;

    /// The finite element constructed from the string
    dealii::SmartPointer<const dealii::FiniteElement<dim>, AmandusApplicationSparse<dim> > fe;

    /// The object handling the degrees of freedom
    dealii::DoFHandler<dim> dof_handler;

    /**
     * @brief The masks used to set boundary conditions, indexed by the
     * boundary indicator
     */
    std::vector<dealii::ComponentMask> boundary_masks;
    
    /// The object holding the constraints for the active mesh
    dealii::ConstraintMatrix     constraints;
  
    dealii::SparsityPattern      sparsity;
    dealii::SparseMatrix<double> matrix;
    const bool use_umfpack;
    dealii::SparseDirectUMFPACK inverse;
    
    dealii::BlockVector<double>  estimates;
};


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
 * The purpose of this class is not so much implementing an
 * application program, but providing the data structures used by all
 * application programs with the following characteristics (first two
 * are inherited from base class):
 * <ol>
 * <li>Single vector and single matrix, no block structures; this does not exclude systems.</li>
 * <li>A single sparse matrix and accordingly single sparse matrices for for each level.</li>
 * <li>A multigrid smoother based on the inversion of vertex patches.</li>
 * </ol>
 *
 * @todo: Straighten up the interface, make private things private
 *
 * @todo: Create interface for ParameterHandler to set parameters for
 * #control, possibly select solver and other parameters
 *
 * @author Guido Kanschat
 * @date 2014
 *
 * @ingroup apps
 */
template <int dim>
class AmandusApplication
  : public AmandusApplicationSparse<dim>
{
  public:
    typedef dealii::MeshWorker::IntegrationInfo<dim> CellInfo;
  
    /**
     * Constructor, setting the finite element and the
     * triangulation. This constructor does not distribute the degrees
     * of freedom or initialize sparsity patterns, which has to be
     * achieved by calling setup_system().
     */
    AmandusApplication(dealii::Triangulation<dim>& triangulation,
		       const dealii::FiniteElement<dim>& fe);

    /**
     * Initialize the finite element system on the current mesh.  This
     * involves distributing degrees of freedom for the leaf mesh and
     * the levels as well as setting up sparsity patterns for the matrices.
     *
     * @note This function calls the virtual function setup_constraints().
     */
    void setup_system ();
  
  public:
    /**
     * Set up hanging node constraints for leaf mesh and for level
     * meshes. Use redefinition in derived classes to add boundary
     * constraints.
     */
    virtual void setup_constraints ();
    /**
     * Use the integrator to build the matrix for the level meshes. This
     * also automatically generates the transfer matrices needed for
     * multigrid with local smoothing on locally refined meshes.
     */
    void assemble_mg_matrix (const dealii::AnyData &in,
			     const AmandusIntegrator<dim>& integrator);
    
    /**
     * Solve the linear system stored in #matrix with the right hand
     * side given. Uses the multigrid preconditioner.
     */
    void solve (dealii::Vector<double>& sol, const dealii::Vector<double>& rhs);
    
    //  protected:

    dealii::MGConstrainedDoFs    mg_constraints;
  
    dealii::MGLevelObject<dealii::SparsityPattern> mg_sparsity;
    dealii::MGLevelObject<dealii::SparseMatrix<double> > mg_matrix;
  
    dealii::MGLevelObject<dealii::SparseMatrix<double> > mg_matrix_down;
    dealii::MGLevelObject<dealii::SparseMatrix<double> > mg_matrix_up;
  
    dealii::MGTransferPrebuilt<dealii::Vector<double> > mg_transfer;
};

/// Compatibility definition
#define AmandusApplicationSparseMultigrid AmandusApplication

/**
 * The same as AmandusApplicationSparse, but with multigrid constraints
 * and homogeneous Dirichlet boundary conditions.
 *
 * @ingroup apps
 */
template <int dim>
class AmandusUMFPACK : public AmandusApplicationSparse<dim>
{
  public:
    AmandusUMFPACK(dealii::Triangulation<dim>& triangulation,
		   const dealii::FiniteElement<dim>& fe);
  private:
    virtual void setup_constraints ();
};


/**
 * A residual operator using AmandusApplicationSparse::assemble_right_hand_side().
 *
 * @ingroup apps
 */
template <int dim>
class AmandusResidual
  : public dealii::Algorithms::Operator<dealii::Vector<double> >
{
  public:
    AmandusResidual(const AmandusApplicationSparse<dim>& application,
		    AmandusIntegrator<dim>& integrator);
		    
    virtual void operator() (dealii::AnyData &out, const dealii::AnyData &in);
  private:
    dealii::SmartPointer<const AmandusApplicationSparse<dim>, AmandusResidual<dim> > application;
    dealii::SmartPointer<AmandusIntegrator<dim>, AmandusResidual<dim> > integrator;
};

/**
 * A solution operator using AmandusApplicationSparse::solve().
 *
 * @ingroup apps
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
    AmandusSolve(AmandusApplicationSparse<dim>& application,
		 AmandusIntegrator<dim>& integrator);
    /**
     * Apply the solution operator. If indecated by events, reassemble matrices 
     */
    virtual void operator() (dealii::AnyData &out, const dealii::AnyData &in);
  private:
    /// The pointer to the application object.
    dealii::SmartPointer<AmandusApplicationSparse<dim>, AmandusSolve<dim> > application;
    /// The pointer to the local integrator for assembling matrices
    dealii::SmartPointer<AmandusIntegrator<dim>, AmandusSolve<dim> > integrator;
};


template <int dim>
inline const dealii::DoFHandler<dim>&
AmandusApplicationSparse<dim>::dofs () const
{
  return dof_handler;
}


#endif
