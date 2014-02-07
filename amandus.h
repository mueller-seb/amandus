/**********************************************************************
 * $Id: cochain.h 1397 2014-02-02 22:42:29Z kanschat $
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


template <int dim>
class AmandusApplication : public dealii::Algorithms::Operator<dealii::Vector<double> >
  
{
  public:
  typedef dealii::MeshWorker::IntegrationInfo<dim> CellInfo;
    
    AmandusApplication(dealii::Triangulation<dim>& triangulation,
		const dealii::FiniteElement<dim>& fe,
		const dealii::MeshWorker::LocalIntegrator<dim>& matrix_integrator,
		const dealii::MeshWorker::LocalIntegrator<dim>& rhs_integrator);

				     /**
				      * Initialize the vector <code>v</code> to the
				      * size matching the DoFHandler.
				      */
    void setup_vector (dealii::Vector<double>& v) const;
    
    /**
     * Initialize the finite element system on the current mesh.
     */
    void setup_system ();

				     /**
				      * Fill the vector
				      * <code>rhs</code> with the
				      * discrete right hand side of
				      * the problem.
				      */
    void assemble_right_hand_side(dealii::Vector<double>& rhs);
    void assemble_right_hand_side(const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
				  dealii::NamedData<dealii::Vector<double> *> &out,
				  const dealii::NamedData<dealii::Vector<double> *> &in) const;
    
				     /**
				      * The solution operator which is
				      * consistent with other algorithms.
				      */
    virtual void operator() (dealii::NamedData<dealii::Vector<double> *> &out,
			     const dealii::NamedData<dealii::Vector<double> *> &in);
    
    void refine_mesh (const bool global = false);
    
  public:
    void setup_constraints ();
    void assemble_matrix (const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
			  dealii::NamedData<dealii::Vector<double> *> &out,
			  const dealii::NamedData<dealii::Vector<double> *> &in);
    void assemble_mg_matrix (const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
			     dealii::NamedData<dealii::Vector<double> *> &out,
			     const dealii::NamedData<dealii::Vector<double> *> &in);
    double estimate(const dealii::MeshWorker::LocalIntegrator<dim>& integrator);
    void error (const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
		const dealii::NamedData<dealii::Vector<double> *> &in,
		unsigned int num_errs);
    void solve (dealii::Vector<double>& sol, const dealii::Vector<double>& rhs);
    void output_results(unsigned int refinement_cycle,
			const dealii::NamedData<dealii::Vector<double>*>* data = 0) const;

    void verify_residual(const dealii::MeshWorker::LocalIntegrator<dim>& integrator,
			 dealii::NamedData<dealii::Vector<double> *> &out,
			 const dealii::NamedData<dealii::Vector<double> *> &in) const;
    
    dealii::ReductionControl control;
//  protected:
    dealii::Triangulation<dim>&       triangulation;
    const dealii::MappingQ1<dim>      mapping;
    const dealii::FiniteElement<dim>& fe;
    dealii::MGDoFHandler<dim>         mg_dof_handler;
    dealii::DoFHandler<dim>&          dof_handler;

    const dealii::MeshWorker::LocalIntegrator<dim>& matrix_integrator;
    const dealii::MeshWorker::LocalIntegrator<dim>& rhs_integrator;

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


template <int dim>
class AmandusResidual
  : public dealii::Algorithms::Operator<dealii::Vector<double> >
{
  public:
    AmandusResidual(const AmandusApplication<dim>& application,
		    const dealii::MeshWorker::LocalIntegrator<dim>& integrator);
		    
    virtual void operator() (dealii::NamedData<dealii::Vector<double> *> &out,
			     const dealii::NamedData<dealii::Vector<double> *> &in);
  private:
    dealii::SmartPointer<const AmandusApplication<dim>, AmandusResidual<dim> > application;
    dealii::SmartPointer<const dealii::MeshWorker::LocalIntegrator<dim>, AmandusResidual<dim> > integrator;
};


template <int dim>
class AmandusSolve
  : public dealii::Algorithms::Operator<dealii::Vector<double> >
{
  public:
    AmandusSolve(AmandusApplication<dim>& application,
		 const dealii::MeshWorker::LocalIntegrator<dim>& integrator);
		    
    virtual void operator() (dealii::NamedData<dealii::Vector<double> *> &out,
			     const dealii::NamedData<dealii::Vector<double> *> &in);
  private:
    dealii::SmartPointer<AmandusApplication<dim>, AmandusSolve<dim> > application;
    dealii::SmartPointer<const dealii::MeshWorker::LocalIntegrator<dim>, AmandusSolve<dim> > integrator;
};


#endif
