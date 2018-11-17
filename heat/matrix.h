/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __matrix_heat_h
#define __matrix_heat_h

/*
 * \file
 * \brief The local integrators for the Heat equation
 * \ingroup Heatgroup
 */
#include <amandus/integrator.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/integration_info.h>
#include <amandus/heat/testing.h> //HELPING FUNCTIONS FOR CODE TESTING

using namespace dealii;
using namespace LocalIntegrators;

/**
 * Integrator for Heat equation.
 *
 * @ingroup integrators
 */
namespace HeatIntegrators
{
template <int dim>
class Conductivity : public Function<dim>
{
public:
  Conductivity(const double margin = 0.0);
  virtual double value(const Point<dim>& p, const unsigned int component = 0) const override;
  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component = 0) const override;

  private:
    const double margin; //MARGIN between low dimensional embedding/pole and boundaries
};

template <int dim>
Conductivity<dim>::Conductivity(const double margin) : margin(margin)
{
}

template <int dim>
double Conductivity<dim>::value(const Point<dim>& p, const unsigned int component) const
{
  double x = p(0);
  double y = p(1);
  bool onEmbedding = (abs(y) < 1e-5) && (abs(x) <= (1-margin));
  double result = 1e-3; //not on face

  if (component == 1) //on face
    {
    if (onEmbedding)
      result = 1e-3; //on embedding
    else
      result = 0; //not on embedding
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
  /**
   * \brief The bilinear form of the Heat equation on the domain
   */
  virtual void cell(MeshWorker::DoFInfo<dim>& dinfo, MeshWorker::IntegrationInfo<dim>& info) const override;
  /**
   * \brief The weak implementation of Dirichlet boundary conditions (vanishing).
  */
  virtual void boundary(MeshWorker::DoFInfo<dim>& dinfo,
                        MeshWorker::IntegrationInfo<dim>& info) const override;
  /**
   * \brief The bilinear form of Heat equation on the low dimensional embedding
  */
  virtual void face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                    MeshWorker::IntegrationInfo<dim>& info1,
                    MeshWorker::IntegrationInfo<dim>& info2) const override;
};

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
            Mii += dx * (fe.shape_grad_component(i,k,d) * fe.shape_grad_component(i,k,d));
         M(i,i) += Mii;
   
         for (unsigned int j=i+1; j<n_dofs; ++j)
	 {
            double Mij = 0.0;
            for (unsigned int d=0; d<n_components; ++d)
               Mij += dx * (fe.shape_grad_component(j,k,d) * fe.shape_grad_component(i,k,d));
            M(i,j) += Mij;
            M(j,i) += Mij;
         }
      }
   }
}

/*Boundary term of Green's formula in weak formulation vanishes due to shape functions v in H_0^1.
  Boundary term of Green's formular (partial integration) occurs in nitsche matrix.*/
template <int dim>
void Matrix<dim>::boundary(MeshWorker::DoFInfo<dim>& dinfo,
                      typename MeshWorker::IntegrationInfo<dim>& info) const
{
/*SOURCE
  if (info.fe_values(0).get_fe().conforms(FiniteElementData<dim>::H1))
  return;

  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  Laplace::nitsche_matrix(dinfo.matrix(0, false).matrix,
                          info.fe_values(0),
                          Laplace::compute_penalty(dinfo, dinfo, deg, deg));*/
}

template <int dim>
void Matrix<dim>::face(MeshWorker::DoFInfo<dim>& dinfo1, MeshWorker::DoFInfo<dim>& dinfo2,
                  MeshWorker::IntegrationInfo<dim>& info1,
                  MeshWorker::IntegrationInfo<dim>& info2) const
{
/* FOR DG WITH INTERIOR PENALTY
if (info1.fe_values(0).get_fe().conforms(FiniteElementData<dim>::H1))
    return;*/

/*SOURCE
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

const unsigned int n_dofs = fe1.dofs_per_cell;
const unsigned int n_comp = fe1.get_fe().n_components();

AssertDimension(2, dim); //essential for 2d cross product
Assert (M1.m() == n_dofs, ExcDimensionMismatch(M1.m(), n_dofs));
Assert (M1.n() == n_dofs, ExcDimensionMismatch(M1.n(), n_dofs));

double ydiff = fe1.quadrature_point(fe1.n_quadrature_points-1)(1)-fe1.quadrature_point(0)(1);
if (abs(ydiff) < 1e-5) //involve horizontal faces only
{
//SOURCE: Laplace::nitsche_matrix() and Laplace::nitsche_tangential_matrix() in laplace.h

   for (unsigned int k=0; k<fe1.n_quadrature_points; ++k)
   {
      const Tensor<1,dim> n = fe1.normal_vector(k);
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
               M1(i,j) += dx * dtu_dot_dtv;
         }
      }
   }

/* 1st ATTEMPT OF EVALUATING SHAPE FUNCTIONS AT THE ENDING POINTS OF THE EMBEDDING (doesn't work properly)
const FiniteElement<dim>& fel1 = info1.finite_element();
const FiniteElement<dim>& fel2 = info2.finite_element();
const double x = 0.5;
const double y = 0;
const Point<dim> p1(-x, y);
const Point<dim> p2(x, y);

const Point<dim>& p12 = fe1.get_mapping().transform_real_to_unit_cell(dinfo1.face, p1);
const Point<dim>& p22 = fe1.get_mapping().transform_real_to_unit_cell(dinfo1.face, p2);
for (unsigned int i=0; i<n_dofs; ++i)
	for (unsigned int j=0; j<n_dofs; ++j)
	{
        const Tensor<1,dim> n = fe1.normal_vector(0);
        Tensor<1,dim> t = cross_product_2d(n);
        t = (1/t.norm())*t;
	M1(i,j) += + kappa.value(p12, 1) * (t*fel1.shape_grad(i, p12)) * fel1.shape_value(j, p12);
	M1(i,j) += - kappa.value(p22, 1) * (t*fel1.shape_grad(i, p22)) * fel1.shape_value(j, p22);
	}*/

/* PRINT QUADRATURE POINTS OF CURRENT FACE (FOR TESTING)
QPointsOut qptsout(fe1, fe2);
qptsout.write();
*/

/* HOW TO PRINT VECTORS (FOR TESTING)
std::ostringstream message;
message << tensor << std::endl;
deallog << message.str();*/
}
}

#endif
