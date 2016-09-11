/**********************************************************************
 *  Copyright (C) 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 *
 *  * <h3> About this example</h3>
 *
 * This example is the easiest program you can write in Amandus.
 * It shows how Amandus work in practice and it gives a brief description
 * of the functions used. It doesn't discuss in detail any individual function
 * but it wants to give a big picture of how things work together.
 * If you wants to learn deeply how a single function work, you can search inside
 * the manual. For informations about deal.II functions and classes, use
 * <a href="https://www.dealii.org/8.4.0/doxygen/deal.II/"> deal.II
 * manual </a>. As concerns the Amandus manual, it is not complete and it is still
 * in development.
 *
 *
 * @ingroup Laplacegroup
 *
 *  <h3> How to compile and run </h3>
 *
 *  Go to <code>example/laplace</code> directory and type:
 * @code
 * make laplace_laplace
  @endcode
 * Now you can see the executable. To execute it and produce the output, type:
 * @code
 * ./laplace
 * @endcode
 *
 *
 * <h3> Introduction </h3>
 *
 * We solve a simple version of Poisson's equation with nonzero right hand side:
 *
 * \f{align*}
 * - \Delta u & = f \qquad\qquad & \text{in}\ \Omega
 * \f}
 *
 * We solve this equation on the unit square, \f$\Omega=[0,1]^2\f$, for which
 * we already know how to generate a mesh from dealii documentation. In
 * this program, we also only consider the particular case \f$f(\mathbf x)=1\f$.
 *
 * We choose the following Cauchy boundary conditions:
 *
 * \f{align*}
 * u &=-1 \qquad\qquad & \text{on}\ x=0,
 * \\
 * u &= 1 \qquad\qquad & \text{on}\ x=1,
 * \\
 * u &= 0 \qquad\qquad &  otherwise.
 * \f}
 *
 * From the basics of the finite element method, you remember the steps we need
 * to take to approximate the solution \f$u\f$ by a finite dimensional approximation.
 * Multiplying from the left by the test function \f$\varphi \f$, integrating over
 * the domain \f$\Omega\f$ and integrating by parts, we get the following weak
 * formulation:
 *
 *
 * @f{align*}
 * \int_\Omega \nabla\varphi \cdot \nabla u
 * -
 * \int_{\partial\Omega} \varphi \mathbf{n}\cdot \nabla u
 *  = \int_\Omega \varphi f.
 * @f}
 *
 * We build a discreate approximation of u:
 *
 * \f{align*}
 * u_h(\mathbf x)=\sum_j U_j \varphi_j(\mathbf x)
 * \f}
 *
 * where \f$\varphi_i(\mathbf x)\f$ are the finite element shape functions we will use.
 *
 * The weak form of the discrete problem is : Find a function \f$u_h\f$, i.e. find
 * the expansion coefficients \f$U_i\f$ mentioned above, so that
 * @f{align*}
 *  (\nabla\varphi_i, \nabla u_h)
 *   = (\varphi_i, f),
 *   \qquad\qquad
 *   i=0\ldots N-1.
 * @f}
 * This equation can be rewritten as a linear
 * system by inserting the representation \f$u_h(\mathbf x)=\sum_j U_j
 * \varphi_j(\mathbf x)\f$: Find a vector \f$U\f$ so that
 * @f{align*}
 *  A U = F,
 * @f}
 * where the matrix \f$A\f$ and the right hand side \f$F\f$ are defined as
 * @f{align*}
 *  A_{ij} &= (\nabla\varphi_i, \nabla \varphi_j),
 *  \\
 *  F_i &= (\varphi_i, f).
 * @f}
 * If you need more details about how to construct a linear system starting from the Poisson's
 * equation, you might be interested in
 * <a href="http://www.dealii.org/developer/doxygen/deal.II/step_3.html"> step-3 </a>
 * of the deal.II tutorial.
 *
 * Now we can go forward and describe how this quantities can be
 * computed with Amandus.
 * @ingroup Laplacegroup
  */

#include <amandus/apps.h>
#include <amandus/laplace/matrix.h>
#include <amandus/laplace/matrix_factor.h>
#include <amandus/laplace/noforce.h>
#include <amandus/laplace/polynomial.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <deal.II/base/function.h>
#include <deal.II/algorithms/newton.h>

template <int dim>
class Startup : public dealii::Function<dim>
{
public:
  Startup();
  virtual double value(const Point<dim>& p, const unsigned int component) const;

  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component) const;
};

// constructor defould one component
template <int dim>
Startup<dim>::Startup()
  : Function<dim>()
{
}

template <int dim>
double
Startup<dim>::value(const Point<dim>& p, const unsigned int) const
{
  double result = 1. * p(0);
  return result;
}

template <int dim>
void
Startup<dim>::value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                         const unsigned int) const
{
  AssertDimension(points.size(), values.size());

  for (unsigned int k = 0; k < points.size(); ++k)
  {
    const Point<dim>& p = points[k];
    values[k] = 1. * p(0);
  }
}

/**
 * We fix the dimension of the domain
 * @code
 * const unsigned int d=2;
 * @endcode
 * Construction of a <code>ofstream</code> and the attachment of
 * the stream to the open file <code>deallog</code>
 * @code
 * std::ofstream logfile("deallog");
 * deallog.attach(logfile);
 * @endcode
 * Now we define an object for a triangulation and we fill
 * it with a single cell of a square domain. The triangulation is
 * refined 3 times, to yield \f$4^3=64\f$ cells in total.
 * @code
 * Triangulation<d> tr;
 * GridGenerator::hyper_cube (tr, -1, 1);
 * tr.refine_global(3);
 * @endcode
 * We construct the tensor product polynomials of degree p. We use
 * continuous polynomials.
 * @code
 * const unsigned int degree = 2;
 * FE_Q<d> fe(degree);
 * @endcode
 * Creation of a Matrix object. The Matrix class permits to construct the
 * matrix A using LocalIntegrators functions. It is essential to implement
 * a class matrix for every kind of problem you want to solve. For more
 * details, read the documentation for the matrix class of the Laplace
 * problem.
 * @code
 * LaplaceIntegrators::Matrix<d> matrix_integrator;
 * @endcode
 *
 * @todo The code does't compile. Correct it and finish the code description.
 */
int
main()
{
  const unsigned int d = 2;

  std::ofstream logfile("deallog");
  deallog.attach(logfile);

  Triangulation<d> tr;
  GridGenerator::hyper_cube(tr, -1, 1);
  tr.refine_global(3);

  const unsigned int degree = 2;
  FE_Q<d> fe(degree);

  Startup<d> startup;
  // std::set<unsigned int> boundaries;
  // boundraies.insert(0);
  // boundaries.insert(1);

  LaplaceIntegrators::Matrix<d> matrix_integrator;

  LaplaceIntegrators::NoForceRHS<d> rhs_integrator;

  AmandusUMFPACK<d> app(tr, fe);
  AmandusSolve<d> solver(app, matrix_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);

  Algorithms::DoFOutputOperator<Vector<double>, d> newout;
  newout.initialize(app.dofs());

  Algorithms::Newton<Vector<double>> newton(residual, solver);
  // newton.parse_parameters(param);

  newton.initialize(newout);
  newton.debug_vectors = true;

  const AmandusIntegrator<d>* AmandInt = 0;
  global_refinement_nonlinear_loop(2, app, newton, AmandInt, AmandInt, &startup);
}
