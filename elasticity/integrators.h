

namespace Elasiticity
{
  template <int dim>
  double elasticity_contract(const Tensor<2,dim>& e1,
			     const Tensor<2,dim>& e2,
			     const double Poisson)
  {
    AssertDimension(dim, 2);
    double result = 0.;
    
    for (unsigned int d1=0;d1<dim;++d1)
      {
	result += e1[d1][d1]*e2[d1][d1];
	for (unsigned int d2=d1+1;d2<dim;++d2)
	  {
	    result += nu * (e1[d1][d1] * e2[d2][d2] +
			    e2[d1][d1] * e1[d2][d2]);
	    result += (1.-nu)/2. * e1[d1][d2] * e2[d1][d2];
	  }
      }
    return result;
  }
  
  
  /**
   * The residual resulting from the linear strain-stress relationship
   * (plane strain)
   * \f{
   * \left[ \begin{array}{c}
   * \sigma_x \\ \sigma_y \\ \tau_{xy}
   * \end{array}\right]
   * = \frac{E}{1-\nu^2}
   * \left[ \begin{array}{ccc}
   * 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2}
   * \end{array}\right]
   * \left[ \begin{array}{c}
   * \epsilon_x \\ \epsilon_y \\ \gamma_{xy}
   * \end{array}\right]
   * \f}
   * with finite deformations, such that
   * \f{
   * \left[ \begin{array}{c}
   * \epsilon_x \\ \epsilon_y \\ \gamma_{xy}
   * \end{array}\right]
   * =
   * \left[ \begin{array}{c}
   * u_x +  \\ \epsilon_y \\ \gamma_{xy}
   * \end{array}\right]
   */
  template <int dim, typename number>
  inline void
  Hooke_finite_strain_residual(
    Vector<number> &result,
    const FEValuesBase<dim> &fe,
    const VectorSlice<const std::vector<std::vector<Tensor<1,dim> > > > &input,
    double Poisson = 0.,
    double Young = 1.)
    {
      const unsigned int nq = fe.n_quadrature_points;
      const unsigned int n_dofs = fe.dofs_per_cell;
      AssertDimension(fe.get_fe().n_components(), dim);

      AssertVectorVectorDimension(input, dim, fe.n_quadrature_points);
      Assert(result.size() == n_dofs, ExcDimensionMismatch(result.size(), n_dofs));

      for (unsigned int k=0; k<nq; ++k)
        {
          const double dx = Young / (1.-Poisson*Poisson) * fe.JxW(k);
	  Tensor<2,dim> Du;
	  for (unsigned int d1=0; d1<dim; ++d1)
	    for (unsigned int d2=d1; d2<dim; ++d2)
	    {
	      Du[d1][d2] = .5 * (input[d1][k][d2] + input[d2][k][d1]);
	      for (unsigned int dd=0; dd<dim; ++dd)
		Du[d1][d2] += .5* input[dd][k][d1] * input[dd][k][d2];
	    }
	  
          for (unsigned int i=0; i<n_dofs; ++i)
	    {
	      Tensor<2,dim> Dv;
	      for (unsigned int d1=0; d1<dim; ++d1)
		for (unsigned int d2=d1; d2<dim; ++d2)
		  {
		    Dv[d1][d2] = .5 * (fe.shape_grad_component(i,k,d1)[d2] + fe.shape_grad_component(i,k,d2)[d1]);
		    for (unsigned int dd=0; dd<dim; ++dd)
		      Dv[d1][d2] += .5 * (fe.shape_grad_component(i,k,dd)[d1] * fe.shape_grad_component(i,k,dd)[d2]);
		  }
	      
	      result(i) += dx * elasticity_contract(Du, Dv, Poisson);
	    }
        }
    }
}
