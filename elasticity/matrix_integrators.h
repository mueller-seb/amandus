namespace Elasticity
{
namespace StVenantKirchhoff
{
/**
 * Gateaux derivative matrix for nonlinear Elasticity problem.
 *
 * \f[
 * \int_\Omega \left \{\big(I+ \nabla v \big) \,
 * \big( \lambda \, \operatorname{tr}(\tfrac12 E) \, I  + \mu E \big) \, \,
 * + \, \, \big(I + \nabla u \big) \,
 * \big(\lambda \,  \operatorname{tr} (\tfrac12 (\,\nabla v + (\nabla v)^T +
 * (\nabla u)^T \nabla v + (\nabla v)^T \nabla u \,)) \,  I
 * + \mu \, (\, \nabla v + (\nabla v)^T +
 * (\nabla u)^T \nabla v + (\nabla v)^T \nabla u \,) \big) \right \} \, : \,  \nabla v \, dx
 * \f]
 *
 * where \f$E = \tfrac12\bigl(\nabla u + (\nabla u)^T + (\nabla u)^T \nabla u \bigr)\f$.
 */

template <int dim>
inline void
cell_matrix(
  dealii::FullMatrix<double>& M, const dealii::FEValuesBase<dim>& fe,
  const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1, dim>>>>& input,
  double lambda = 0., double mu = 1.)
{
  const unsigned int nq = fe.n_quadrature_points;
  const unsigned int n_dofs = fe.dofs_per_cell;

  for (unsigned int k = 0; k < nq; ++k)
  {
    const double dx = fe.JxW(k);
    // F(u) = (I+grad u)
    dealii::Tensor<2, dim> F;
    // E builded as in parameters.h
    dealii::Tensor<2, dim> E;
    for (unsigned int d1 = 0; d1 < dim; ++d1)
    {
      F[d1][d1] = 1.;
      for (unsigned int d2 = 0; d2 < dim; ++d2)
      {
        F[d1][d2] += input[d1][k][d2];
        E[d1][d2] = .5 * (input[d1][k][d2] + input[d2][k][d1]);
        for (unsigned int dd = 0; dd < dim; ++dd)
          E[d1][d2] += .5 * (input[dd][k][d1] * input[dd][k][d2]);
      }
    }

    double trace = 0.;
    for (unsigned int dd = 0; dd < dim; ++dd)
      trace += E[dd][dd];
    for (unsigned int dd = 0; dd < dim; ++dd)
      E[dd][dd] += lambda / (2. * mu) * trace;

    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      // build F_i
      dealii::Tensor<2, dim> F_i;
      // build E_i
      dealii::Tensor<2, dim> E_i;
      for (unsigned int d1 = 0; d1 < dim; ++d1)
      {
        F_i[d1][d1] = 1.;
        for (unsigned int d2 = 0; d2 < dim; ++d2)
        {
          F_i[d1][d2] += fe.shape_grad_component(i, k, d1)[d2];
          E_i[d1][d2] =
            .5 * (fe.shape_grad_component(i, k, d1)[d2] + fe.shape_grad_component(i, k, d2)[d1]);

          double Ed1d2 = 0.;
          for (unsigned int dd = 0; dd < dim; ++dd)
            //.5 * (grad u)^T * grad v
            Ed1d2 += .5 * (input[dd][k][d1] * fe.shape_grad_component(i, k, dd)[d2]);

          E_i[d1][d2] += Ed1d2;
          E_i[d2][d1] += Ed1d2;
        }
      }

      double trace = 0.;
      for (unsigned int dd = 0; dd < dim; ++dd)
        trace += E_i[dd][dd];
      for (unsigned int dd = 0; dd < dim; ++dd)
        E_i[dd][dd] += lambda / (2. * mu) * trace;

      // left term computation
      // L_i: left tensor
      dealii::Tensor<2, dim> L_i;

      // F_i * E
      for (unsigned int d1 = 0; d1 < dim; ++d1)
        for (unsigned int d2 = 0; d2 < dim; ++d2)
        {
          double temp = 0.;
          for (unsigned int dd = 0; dd < dim; ++dd)
          {
            L_i[d1][d2] += F_i[d1][dd] * E[dd][d2];
            temp += F[d1][dd] * E_i[dd][d2];
          }
          L_i[d1][d2] += temp;
        }

      //: product
      for (unsigned int j = 0; j < n_dofs; ++j)
      {
        double temp = 0.;
        for (unsigned int d1 = 0; d1 < dim; ++d1)
          for (unsigned int d2 = 0; d2 < dim; ++d2)
            temp += fe.shape_grad_component(j, k, d1)[d2] * L_i[d1][d2];

        M(i, j) += 2. * mu * dx * temp;
      }
    }
  }
}
}
}
