#ifndef __cahn_hilliard_residual_h
#define __cahn_hilliard_residual_h

#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/advection.h>
#include <integrator.h>

namespace CahnHilliard
{
  using namespace dealii;
  using namespace LocalIntegrators;
  using namespace MeshWorker;
  
  /**
   * Local integrator for the residual of a stationary Cahn-Hilliard
   * problem. 
   * \f[
   * \vvect{v}{w} \mapsto 
   * \begin{pmatrix}
   * (\sigma, v) - (W'(u), v) - \epsilon^2(\nabla u, \nabla v) \\
   * (\nabla \sigma, \nabla w)
   * \end{pmatrix}
   * \f]
   */
  template <int dim>
    class Residual : public AmandusIntegrator<dim>
  {
    public:
      Residual(double diffusion, const Function<dim>& advection);

      virtual void cell(DoFInfo<dim>& dinfo,
                        IntegrationInfo<dim>& info) const;

    private:
      const double diffusion;
      const Function<dim>* const advection;
  };


  template <int dim>
    Residual<dim>::Residual(double diffusion, const Function<dim>& advection) :
      diffusion(diffusion), advection(&advection)
    {
      this->use_cell = true;
      this->use_boundary = false;
      this->use_face = false;
    }


  template <int dim>
    void Residual<dim>::cell(
        DoFInfo<dim>& dinfo, 
        IntegrationInfo<dim>& info) const
    {
      Assert(info.values.size() >= 1,
             ExcDimensionMismatch(info.values.size(), 1));
      Assert(info.gradients.size() >= 1,
             ExcDimensionMismatch(info.values.size(), 1));

      const unsigned int n_qpoints = info.fe_values(1).n_quadrature_points;
      std::vector<std::vector<double> > direction(
          dim, std::vector<double>(n_qpoints));
      this->advection->vector_values(info.fe_values(1).get_quadrature_points(),
                                     direction);

      const std::vector<std::vector<double> >& point = info.values[0];
      const std::vector<std::vector<Tensor<1, dim> > >& Dpoint = info.gradients[0];

      std::vector<double> minus_d_potential(point[1].size());
      for(unsigned int q = 0; q < minus_d_potential.size(); ++q)
      {
        minus_d_potential[q] = (
            point[1][q] * (1.0 - point[1][q] * point[1][q])) / diffusion;
      }

      L2::L2(dinfo.vector(0).block(0),
             info.fe_values(0), point[0]);
      L2::L2(dinfo.vector(0).block(0),
             info.fe_values(0), minus_d_potential);
      Laplace::cell_residual(dinfo.vector(0).block(0),
                             info.fe_values(0),
                             Dpoint[1],
                             -1.0*diffusion);

      Laplace::cell_residual(dinfo.vector(0).block(1),
                             info.fe_values(1),
                             Dpoint[0]);

      dealii::LocalIntegrators::Advection::cell_residual(
          dinfo.vector(0).block(1),
          info.fe_values(1),
          Dpoint[1],
          direction
          );
    }
}

#endif
