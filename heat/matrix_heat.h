/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __matrix_heat_heat_h
#define __matrix_heat_heat_h

#include <amandus/integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/integration_info.h>

using namespace dealii;
using namespace LocalIntegrators;

/**
 * Integrator for heat equation.
 *
 * The distinction between stationary and instationary problems is
 * made by the variable AmandusIntegrator::timestep, which is
 * inherited from the base class. If this variable is zero, we solve a
 * stationary problem. If it is nonzero, we assemble for an implicit
 * scheme.
 */
namespace HeatIntegrators
{
template <int dim>
class Conductivity : public dealii::Function<dim>
{
public:
  Conductivity();
  virtual double value(const Point<dim>& p, const unsigned int component = 0) const;
  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component = 0) const;
};

template <int dim>
Conductivity<dim>::Conductivity() : Function<dim>()
{
}

template <int dim>
double Conductivity<dim>::value(const Point<dim>& p, const unsigned int component) const
{
  double x = p(0);
  double y = p(1);
  double result = 1e-5;
  if (component == 1)
    {
    result = 0;
    if ((abs(y) < 1e-5) && (abs(x) <= 1))
      result = 1;
    }
  return result;
}

template <int dim>
void Conductivity<dim>::value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                         const unsigned int) const
{
  AssertDimension(points.size(), values.size());

  for (unsigned int k = 0; k < points.size(); ++k)
  {
    const Point<dim>& p = points[k];
    values[k] = 1. * p(0);
  }
}

template <int dim>
class Matrix : public AmandusIntegrator<dim>
{
public:
  Matrix();
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo,
                        MeshWorker::IntegrationInfo<dim>& info) const;
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                    MeshWorker::IntegrationInfo<dim>& info1,
                    MeshWorker::IntegrationInfo<dim>& info2) const;
};

/**
 * \brief Integrator for the matrix of the differential operator.
 */

template <int dim>
Matrix<dim>::Matrix()
{
  this->use_boundary = false;
  this->use_face = true;
}

template <int dim>
void Matrix<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const
{
  AssertDimension(dinfo.n_matrices(), 1);

  Conductivity<dim> kappa;
  FullMatrix<double>& M = dinfo.matrix(0, false).matrix;
  const FEValuesBase<dim>& fe = info.fe_values(0);
  const unsigned int n_dofs = fe.dofs_per_cell;
  const unsigned int n_components = fe.get_fe().n_components();

      for (unsigned int k=0; k<fe.n_quadrature_points; ++k)
         {
              const double dx = fe.JxW(k) * kappa.value(fe.quadrature_point(k), 0);
           for (unsigned int i=0; i<n_dofs; ++i)
              {
               double Mii = 0.0;
               for (unsigned int d=0; d<n_components; ++d)
                   Mii += dx *
                           (fe.shape_grad_component(i,k,d) * fe.shape_grad_component(i,k,d));
   
                 M(i,i) += Mii;
   
                 for (unsigned int j=i+1; j<n_dofs; ++j)
{
                     double Mij = 0.0;
                     for (unsigned int d=0; d<n_components; ++d)
                        Mij += dx *
                              (fe.shape_grad_component(j,k,d) * fe.shape_grad_component(i,k,d));
  
                    M(i,j) += Mij;
                    M(j,i) += Mij;
  		 }
        }
   }
}

/*Boundary term of Green's formula diminishes due to shape functions v in H_0^1. Boundary term of Green's formular (partial integration) is part of the nitsche matrix.*/
template <int dim>
void Matrix<dim>::boundary(MeshWorker::DoFInfo<dim>& dinfo,
                      typename MeshWorker::IntegrationInfo<dim>& info) const
{
/*  if (info.fe_values(0).get_fe().conforms(FiniteElementData<dim>::H1))
    return;

  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_matrix(dinfo.matrix(0, false).matrix,
                          info.fe_values(0),
                          Laplace::compute_penalty(dinfo, dinfo, deg, deg));
*/
}

template <int dim>
class QPointsOut
{
private:
	const std::vector<Point<dim>>& qp1;
        const std::vector<Point<dim>>& qp2;
	unsigned int n1 = 0;
	unsigned int n2 = 0;
	const double eps = 1e-5;
	bool y0 = true;
public:
  QPointsOut(const FEValuesBase<dim>& fe1, const FEValuesBase<dim>& fe2);
  void write();
};

template <int dim>
QPointsOut<dim>::QPointsOut(const FEValuesBase<dim>& fe1, const FEValuesBase<dim>& fe2):
qp1(fe1.get_quadrature_points()), qp2(fe2.get_quadrature_points())
{
	n1 = qp1.size();
	n2 = qp2.size();
	Assert((n1 == n2), ExcInternalError());
}


template <int dim>
void QPointsOut<dim>::write()
{
std::vector<double> x1(n1), x2(n2), y1(n1), y2(n2);

for (int i = 0; i < n1; i++) {
	//Point<dim> pt = qp1[i];
	y1[i] = qp1[i](1);
	y2[i] = qp2[i](1);
	x1[i] = qp1[i](0);
	x2[i] = qp2[i](0);
	if ((abs(y1[i])>eps) or (abs(y2[i])>eps))
		{ y0 = false; }
}
if (y0) {
std::ostringstream message;
message << "new face with quadrature points" << std::endl;

for (int i = 0; i < n1; i++) {
message << "(" << x1[i] << ", " << y1[i] << ") - (" << x2[i] << ", " << y2[i] << ")" << std::endl;
}

deallog << message.str();
}
}



template <int dim>
void Matrix<dim>::face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                  MeshWorker::IntegrationInfo<dim>& info1,
                  MeshWorker::IntegrationInfo<dim>& info2) const
{
if (info1.fe_values(0).get_fe().conforms(FiniteElementData<dim>::H2)) //H1 original, H2 due to non-DGQ?
    return;
/*
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  Laplace::ip_matrix(dinfo1.matrix(0, false).matrix,
                     dinfo1.matrix(0, true).matrix,
                     dinfo2.matrix(0, true).matrix,
                     dinfo2.matrix(0, false).matrix,
                     info1.fe_values(0),
                     info2.fe_values(0),
                     Laplace::compute_penalty(dinfo1, dinfo2, deg, deg));*/

Conductivity<dim> kappa;

FullMatrix<double>& M1 = dinfo1.matrix(0, false).matrix;
FullMatrix<double>& M2 = dinfo2.matrix(0, false).matrix;
const FEValuesBase<dim>& fe1 = info1.fe_values(0);
const FEValuesBase<dim>& fe2 = info2.fe_values(0);

/*
QPointsOut qptsout(fe1, fe2);
qptsout.write();
*/

const unsigned int n_dofs = fe1.dofs_per_cell;
const unsigned int n_comp = fe1.get_fe().n_components();
//AssertDimension(fe1.get_fe().n_components(), dim);
Assert (M1.m() == n_dofs, ExcDimensionMismatch(M1.m(), n_dofs));
Assert (M1.n() == n_dofs, ExcDimensionMismatch(M1.n(), n_dofs));

for (unsigned int k=0; k<fe1.n_quadrature_points; ++k)
{
        const Tensor<1,dim> n = fe1.normal_vector(k);
	AssertDimension(2, dim);
        Tensor<1,dim> t = cross_product_2d(n);
        t = (1/t.norm())*t;
	const double dx = fe1.JxW(k)*kappa.value(fe1.quadrature_point(k), 1);
	for (unsigned int i=0; i<n_dofs; ++i)
		for (unsigned int j=0; j<n_dofs; ++j)
		{
		double tgradu = 0.;
		double tgradv = 0.;
		double dtu_dot_dtv = 0.;
		         for (unsigned int d=0; d<n_comp; ++d)
		         {
		             tgradu = t*fe1.shape_grad_component(j,k,d);
		             tgradv = t*fe1.shape_grad_component(i,k,d);
			     dtu_dot_dtv += tgradu * tgradv;
		          } 
                 M1(i,j) += dx * dtu_dot_dtv * 0.5; //negative sign causes error
                 }
}



for (unsigned int k=0; k<fe2.n_quadrature_points; ++k)
{
        const Tensor<1,dim> n = fe1.normal_vector(k);
	AssertDimension(2, dim);
        Tensor<1,dim> t = cross_product_2d(n);
        t = (1/t.norm())*t;
	const double dx = fe2.JxW(k)*kappa.value(fe2.quadrature_point(k), 1);
	for (unsigned int i=0; i<n_dofs; ++i)
		for (unsigned int j=0; j<n_dofs; ++j)
		{
		double tgradu = 0.;
		double tgradv = 0.;
		double dtu_dot_dtv = 0.;
		         for (unsigned int d=0; d<n_comp; ++d)
		         {
		             tgradu = t*fe2.shape_grad_component(j,k,d);
		             tgradv = t*fe2.shape_grad_component(i,k,d);
			     dtu_dot_dtv += tgradu * tgradv;
		          } 
		 M2(i,j) += dx * dtu_dot_dtv * 0.5; //negative sign causes error
                 }
}
}


}


#endif
