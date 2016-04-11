/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#include <deal.II/base/flow_function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>

#include <amandus/amandus.h>
#include <amandus/amandus_arpack.h>

using namespace dealii;


template <int dim>
AmandusUMFPACK<dim>::AmandusUMFPACK(
  Triangulation<dim>& triangulation,
  const FiniteElement<dim>& fe)
		:
		AmandusApplicationSparse<dim>(triangulation, fe, true)
{}


//----------------------------------------------------------------------//

template <int dim>
AmandusResidual<dim>::AmandusResidual(const AmandusApplicationSparse<dim>& application,
				      AmandusIntegrator<dim>& integrator)
		:
		application(&application),
		integrator(&integrator)
{}


template <int dim>
void
AmandusResidual<dim>::operator() (dealii::AnyData &out, const dealii::AnyData &in)
{
  integrator->extract_data(in);
  application->setup_vector(*out.entry<Vector<double>*>(0));
  *out.entry<Vector<double>*>(0) = 0.;
  application->assemble_right_hand_side(out, in, *integrator);
  
  // if we are assembling the residual for a Newton step within a
  // timestepping scheme, we have to subtract the previous time.
  const Vector<double>* p = in.try_read_ptr<Vector<double> >("Previous time");
  if (p != 0)
    out.entry<Vector<double>*>(0)->add(-1., *p);
}


//----------------------------------------------------------------------//

template <int dim>
AmandusSolve<dim>::AmandusSolve(AmandusApplicationSparse<dim>& application,
				AmandusIntegrator<dim>& integrator)
		:
		application(&application),
		integrator(&integrator)
{}


template <int dim>
void
AmandusSolve<dim>::operator() (dealii::AnyData &out, const dealii::AnyData &in)
{
  integrator->extract_data(in);  
  if (this->notifications.test(Algorithms::Events::initial)
      || this->notifications.test(Algorithms::Events::remesh)
      || this->notifications.test(Algorithms::Events::bad_derivative))
    {
      dealii::deallog << "Assemble matrices" << std::endl;
      application->assemble_matrix(in, *integrator);
      application->assemble_mg_matrix(in, *integrator);
      this->notifications.clear();
    }
  const Vector<double>* rhs = in.read_ptr<Vector<double> >(0);
  Vector<double>* solution = out.entry<Vector<double>*>(0);
  
  application->solve(*solution, *rhs);
//  deallog << "Norms " << rhs->l2_norm() << ' ' << solution->l2_norm() << std::endl;
}

//----------------------------------------------------------------------//

template <int dim>
AmandusArpack<dim>::AmandusArpack(AmandusApplicationSparse<dim>& application,
				AmandusIntegrator<dim>& integrator)
		:
		application(&application),
		integrator(&integrator)
{}


template <int dim>
void
AmandusArpack<dim>::operator() (dealii::AnyData &out, const dealii::AnyData &in)
{
  integrator->extract_data(in);  
  if (this->notifications.test(Algorithms::Events::initial)
      || this->notifications.test(Algorithms::Events::remesh))
    {
      dealii::deallog << "Assemble matrices" << std::endl;
      application->assemble_matrix(in, *integrator, true);
      application->assemble_mg_matrix(in, *integrator);
      this->notifications.clear();
    }

  std::vector<std::complex<double> >* eigenvalues
    = out.entry<std::vector<std::complex<double> >* > ("eigenvalues");
  
  std::vector<Vector<double> >* eigenvectors = out.entry<std::vector<Vector<double> >*>("eigenvectors");
  
  application->arpack_solve(*eigenvalues, *eigenvectors);
}


template class AmandusUMFPACK<2>;
template class AmandusUMFPACK<3>;

template class AmandusResidual<2>;
template class AmandusResidual<3>;

template class AmandusSolve<2>;
template class AmandusSolve<3>;

template class AmandusArpack<2>;
template class AmandusArpack<3>;
