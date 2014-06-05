
#ifndef __brusselator_ode_h
#define __brusselator_ode_h

#include <deal.II/algorithms/operator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/smartpointer.h>
#include <brusselator/parameters.h>

class Explicit
  : public dealii::Algorithms::Operator<dealii::Vector<double> >
{
  public:
    Explicit(const Brusselator::Parameters& par);
    virtual void operator() (dealii::AnyData &out, const dealii::AnyData &in);
    
  private:
    dealii::SmartPointer<const Brusselator::Parameters, class Explicit> parameters;
};


Explicit::Explicit(const Brusselator::Parameters& par)
		:
		parameters(&par)
{}


void
Explicit:: operator() (dealii::AnyData &out, const dealii::AnyData &in)
{
  const double ts = *in.read_ptr<double>("Timestep");
  dealii::Vector<double>& r = *out.entry<dealii::Vector<double>*>(0);
  const dealii::Vector<double>& n = *in.read_ptr<dealii::Vector<double> >("Previous iterate");
  const double u = n(0);
  const double v = n(1);
  r(0) = u - ts * (-parameters->B - u*u*v + (parameters->A+1.)*u );
  r(1) = v - ts * (-parameters->A*u + u*u*v );
}


class ImplicitResidual
  : public dealii::Algorithms::Operator<dealii::Vector<double> >
{
  public:
    ImplicitResidual(const Brusselator::Parameters& par);
    virtual void operator() (dealii::AnyData &out, const dealii::AnyData &in);
    
  private:
    dealii::SmartPointer<const Brusselator::Parameters, class ImplicitResidual> parameters;
};


ImplicitResidual::ImplicitResidual(const Brusselator::Parameters& par)
		:
		parameters(&par)
{}


void
ImplicitResidual:: operator() (dealii::AnyData &out, const dealii::AnyData &in)
{
  const double ts = *in.read_ptr<double>("Timestep");
  dealii::Vector<double>& r = *out.entry<dealii::Vector<double>*>(0);
const dealii::Vector<double>& n = *in.read_ptr<dealii::Vector<double> >("Newton iterate");
const dealii::Vector<double>& p = *in.read_ptr<dealii::Vector<double> >("Previous time");
  const double u = n(0);
  const double v = n(1);
  
  r(0) = u - p(0) + ts * (-parameters->B - u*u*v + (parameters->A+1.)*u );
  r(1) = v - p(1) + ts * (-parameters->A*u + u*u*v );
}


class ImplicitSolve
  : public dealii::Algorithms::Operator<dealii::Vector<double> >
{
  public:
    ImplicitSolve(const Brusselator::Parameters& par);
    virtual void operator() (dealii::AnyData &out, const dealii::AnyData &in);
    
  private:
    dealii::SmartPointer<const Brusselator::Parameters, class ImplicitSolve> parameters;
};


ImplicitSolve::ImplicitSolve(const Brusselator::Parameters& par)
		:
		parameters(&par)
{}


void
ImplicitSolve:: operator() (dealii::AnyData &out, const dealii::AnyData &in)
{
  const double ts = *in.read_ptr<double>("Timestep");
  dealii::Vector<double>& s = *out.entry<dealii::Vector<double>*>(0);
const dealii::Vector<double>& r = *in.read_ptr<dealii::Vector<double> >("Newton residual");
const dealii::Vector<double>& n = *in.read_ptr<dealii::Vector<double> >("Newton iterate");
  const double u = n(0);
  const double v = n(1);

  dealii::deallog << "ImS " << ts << std::endl;
dealii::FullMatrix<double> M(2,2);
M(0,0) = 1. + ts * (2.*u*v - (parameters->A+1.));
M(1,1) = 1. + ts * u*u;
M(0,1) = ts * u*u;
M(1,0) = ts * 2.*u*v;

M.gauss_jordan();
M.vmult(s, r);
}


#endif
