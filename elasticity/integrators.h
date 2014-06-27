

namespace Elasticity
{
  template <int dim>
  double elasticity_contract(const dealii::Tensor<2,dim>& e1,
			     const dealii::Tensor<2,dim>& e2,
			     const double lambda,
			     const double mu)
  {
    Assert(dim==2, dealii::ExcNotImplemented());
    double trace1 = 0.;
    double trace2 = 0.;
    double result = 0.;
    
    for (unsigned int d1=0;d1<dim;++d1)
      {
	trace1 += e1[d1][d1];
	trace2 += e2[d1][d1];
	
 	result += 2.* mu * e1[d1][d1]*e2[d1][d1];
	for (unsigned int d2=d1+1;d2<dim;++d2)
	  {
	    result += 4.* mu * e1[d1][d2] * e2[d1][d2];
	  }
      }

    result += lambda * trace1 * trace2;
    
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
    dealii::Vector<number> &result,
    const dealii::FEValuesBase<dim> &fe,
    const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1,dim> > > > &input,
    double lambda = 0.,
    double mu = 1.)
    {
      const unsigned int nq = fe.n_quadrature_points;
      const unsigned int n_dofs = fe.dofs_per_cell;
      
      // AssertDimension(fe.get_fe().n_components(), dim);
      // AssertVectorVectorDimension(input, dim, fe.n_quadrature_points);
      // Assert(result.size() == n_dofs, ExcDimensionMismatch(result.size(), n_dofs));

      for (unsigned int k=0; k<nq; ++k)
        {
          const double dx = fe.JxW(k);
	  dealii::Tensor<2,dim> Du;
	  for (unsigned int d1=0; d1<dim; ++d1)
	    for (unsigned int d2=d1; d2<dim; ++d2)
	    {
	      Du[d1][d2] = .5 * (input[d1][k][d2] + input[d2][k][d1]);
	      double t = Du[d1][d2];
	      for (unsigned int dd=0; dd<dim; ++dd)
	      	Du[d1][d2] += .5* input[dd][k][d1] * input[dd][k][d2];
//	      dealii::deallog << '[' << (Du[d1][d2]-t)/t << ']';
	    }
	  
          for (unsigned int i=0; i<n_dofs; ++i)
	    {
	      dealii::Tensor<2,dim> Dv;
	      for (unsigned int d1=0; d1<dim; ++d1)
		for (unsigned int d2=d1; d2<dim; ++d2)
		  {
		    Dv[d1][d2] = .5 * (fe.shape_grad_component(i,k,d1)[d2] + fe.shape_grad_component(i,k,d2)[d1]);
		    for (unsigned int dd=0; dd<dim; ++dd)
		      Dv[d1][d2] += .5 * (fe.shape_grad_component(i,k,dd)[d1] * fe.shape_grad_component(i,k,dd)[d2]);
		  }
	      
	      result(i) += dx * elasticity_contract(Du, Dv, lambda, mu);
	    }
        }
    }
}


