
namespace CahnHilliard
{
  using namespace dealii;
  using namespace LocalIntegrators;


  template <int dim>
    class Matrix : public AmandusIntegrator<dim>
  {
    public:
      Matrix();

      virtual void cell(MeshWorker::DoFInfo<dim>& dinfo,
                        MeshWorker::IntegrationInfo<dim>& info) const;
  };


  template <int dim>
    Matrix<dim>::Matrix()
  {
    this->use_cell = true;
    this->use_face = false;
    this->use_boundary = false;
  }


  template <int dim>
    void Matrix<dim>::cell(MeshWorker::DoFInfo<dim>& dinfo,
                           MeshWorker::IntegrationInfo<dim>& info) const
    {
      AssertDimension(dinfo.n_matrices(), 4);
      Assert(info.values.size() > 0,
             ExcDimensionMismatch(info.values.size(), 1));

      // values of the function at which we are linearizing
      const std::vector<double>& point_of_linearization = info.values[0][1];

      // fixed potential function for now
      std::vector<double> minus_dd_potential(point_of_linearization.size());
      for(int q = 0; q < minus_dd_potential.size(); ++q)
      {
        minus_dd_potential[q] = (
            1.0 - 3.0 * point_of_linearization[q]*point_of_linearization[q]);
      }
      // fixed epsilon square for now
      double minus_eps_square = -1e-2;

      L2::mass_matrix(dinfo.matrix(0).matrix,
                      info.fe_values(0));

      // actually, these operators are from fe_values(1) to the dual of
      // fe_values(0), but we assume that they are equal
      L2::weighted_mass_matrix(dinfo.matrix(1).matrix,
                               info.fe_values(1),
                               minus_dd_potential);
      Laplace::cell_matrix(dinfo.matrix(1).matrix,
                           info.fe_values(1),
                           minus_eps_square);

      Laplace::cell_matrix(dinfo.matrix(2).matrix,
                           info.fe_values(0));
    }
}
