#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <fstream>

#include <deal.II/base/logstream.h>

#include <amandus/elasticity/matrix_integrators.h>

#include <amandus/elasticity/integrators.h>

using namespace dealii;

template <int dim>
void
hypercube(Triangulation<dim>& tr, unsigned int refinement = 0, bool local = false)
{
  GridGenerator::hyper_cube(tr, -1., 1.);
  if (refinement && !local)
    tr.refine_global(refinement);
  if (refinement && local)
  {
    tr.refine_global(1);
    for (unsigned int i = 1; i < refinement; ++i)
    {
      for (typename Triangulation<dim>::active_cell_iterator cell = tr.begin_active();
           cell != tr.end();
           ++cell)
      {
        const Point<dim>& p = cell->center();
        bool negative = true;
        for (unsigned int d = 0; d < dim; ++d)
          if (p(d) >= 0.)
            negative = false;
        if (negative)
          cell->set_refine_flag();
      }
      tr.execute_coarsening_and_refinement();
    }
  }
  deallog << "Triangulation hypercube " << dim << "D refinement " << refinement;
  if (local)
    deallog << " local ";
  deallog << " steps " << tr.n_active_cells() << " active cells " << tr.n_cells() << " total cells "
          << std::endl;
}

std::string deallogname;
std::ofstream deallogfile;

void
initlog(bool console = false)
{
  deallogname = "output";
  deallogfile.open(deallogname.c_str());
  deallog.attach(deallogfile);
  deallog.depth_console(console ? 10 : 0);

  // TODO: Remove this line and replace by test_mode()
  deallog.threshold_float(1.e-8);
}

int
main()
{
  const double lambda = 1;
  const double mu = 4;

  const unsigned int dim = 2;
  const unsigned int n_iterations = 10;
  double eps = 0.01;

  initlog();

  Triangulation<dim> tr;
  hypercube(tr, 1);

  FE_Q<dim> q(1);
  FESystem<dim> fe(q, dim);

  QGauss<dim> quadrature(fe.tensor_degree() + 1);
  FEValues<dim> fev(fe, quadrature, update_gradients);
  typename Triangulation<dim>::cell_iterator cell1 = tr.begin(1);
  fev.reinit(cell1);

  const unsigned int n = fev.dofs_per_cell;

  FullMatrix<double> M(n, n);
  Vector<double> u(n), d(n), M_vector(n), u1(n), M_vector1(n);

  for (unsigned int direction1 = 0; direction1 < n; ++direction1)
  {
    deallog << "DIRECTION:   " << direction1 << std::endl;
    d(direction1) = 1;

    std::vector<std::vector<Tensor<1, dim>>> ugrad(
      dim, std::vector<Tensor<1, dim>>(fev.n_quadrature_points));

    std::vector<types::global_dof_index> indices(n);
    for (unsigned int i = 0; i < n; ++i)
    {
      indices[i] = i;
      u[i] = i;
    }

    fev.get_function_gradients(
      u, indices, VectorSlice<std::vector<std::vector<Tensor<1, dim>>>>(ugrad), true);

    Elasticity::StVenantKirchhoff::cell_matrix(M, fev, make_slice(ugrad), lambda, mu);

    M.vmult(M_vector, d);

    Vector<double> residual_u(n);

    Elasticity::StVenantKirchhoff::cell_residual(residual_u, fev, make_slice(ugrad), lambda, mu);

    for (unsigned int i = 0; i < n_iterations; ++i)
    {
      std::vector<std::vector<Tensor<1, dim>>> ugrad1(
        dim, std::vector<Tensor<1, dim>>(fev.n_quadrature_points));

      Vector<double> residual_updir(n);
      M_vector1 = M_vector;
      u1 = u;
      u1(direction1) += eps;
      fev.get_function_gradients(
        u1, indices, VectorSlice<std::vector<std::vector<Tensor<1, dim>>>>(ugrad1), true);
      Elasticity::StVenantKirchhoff::cell_residual(
        residual_updir, fev, make_slice(ugrad1), lambda, mu);

      M_vector1 *= eps;
      residual_updir -= residual_u;
      residual_updir -= M_vector1;

      deallog << "Error:   " << residual_updir.l2_norm() << std::endl;
      eps = eps / 2;
    }
    deallog << std::endl;
  }
}
