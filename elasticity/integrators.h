

namespace Elasticity
{
/**
 * Integrators for linear stress-strain relation in the case of
 * geometrically nonlinear elasticity, also known as
 * St. Venant-Kirchhoff materials or as large displacement/small strain.
 */
namespace StVenantKirchhoff
{
/**
 * The residual for nonlinear Elasticity problem.
 *
 * \f[
 * \int_\Omega \bigl((I+F) \Sigma\bigr) \colon \nabla v \,dx
 * \f]
 *
 * @author Guido Kanschat
 */
template <int dim, typename number>
inline void
cell_residual(
  dealii::Vector<number>& result, const dealii::FEValuesBase<dim>& fe,
  const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1, dim>>>>& input,
  double lambda = 0., double mu = 1.)
{
  const unsigned int nq = fe.n_quadrature_points;
  const unsigned int n_dofs = fe.dofs_per_cell;

  // AssertDimension(fe.get_fe().n_components(), dim);
  // AssertVectorVectorDimension(input, dim, fe.n_quadrature_points);
  // Assert(result.size() == n_dofs, ExcDimensionMismatch(result.size(), n_dofs));

  for (unsigned int k = 0; k < nq; ++k)
  {
    const double dx = fe.JxW(k);
    // Deformation gradient F = I + grad u
    dealii::Tensor<2, dim> F;
    // The Green - St. Venant strain tensor (F^TF - I)/2
    dealii::Tensor<2, dim> E;
    for (unsigned int d1 = 0; d1 < dim; ++d1)
    {
      F[d1][d1] = 1.;
      for (unsigned int d2 = 0; d2 < dim; ++d2)
      {
        F[d1][d2] += input[d1][k][d2];
        E[d1][d2] = .5 * (input[d1][k][d2] + input[d2][k][d1]);
        for (unsigned int dd = 0; dd < dim; ++dd)
          E[d1][d2] += .5 * input[dd][k][d1] * input[dd][k][d2];
      }
    }

    double trace = 0.;
    for (unsigned int dd = 0; dd < dim; ++dd)
      trace += E[dd][dd];
    for (unsigned int dd = 0; dd < dim; ++dd)
      E[dd][dd] += lambda / (2. * mu) * trace;

    // Now that we have all the variables, let's test against
    // the gradient(!) of the test functions

    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      for (unsigned int d1 = 0; d1 < dim; ++d1)
        for (unsigned int d2 = 0; d2 < dim; ++d2)
        {
          double dv = fe.shape_grad_component(i, k, d1)[d2];
          // Compute (F Sigma)_ij (Ciarlet Theorem 2.6.2)

          double stress = 0.;
          for (unsigned int dd = 0; dd < dim; ++dd)
            stress += 2. * mu * F[d1][dd] * E[dd][d2];

          result(i) += dx * stress * dv;
        }
    }
  }
}
}
}
