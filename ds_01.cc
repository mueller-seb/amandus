// $Id$

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include "amandus.h"
#include "matrix_darcy_stokes.h"
#include "rhs_one.h"


int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  std::vector<unsigned int> sub(d,3);
  Point<d> p1;
  Point<d> p2;

  for (unsigned int i=0;i<d;++i)
    {
      p1(i) = -1.;
      p2(i) = 1.;
    }
  
  GridGenerator::subdivided_hyper_rectangle (tr, sub, p1, p2);

  const unsigned int degree = 1;
  FE_RaviartThomas<d> vec(degree);
  FE_DGQ<d> scal(degree);
  FESystem<d> fe(vec, 1, scal, 1);
  
  DarcyStokesMatrix<d> matrix_integrator(1000.);
  RhsOne<d> rhs_integrator;

  AmandusApplication<d> test1(tr, fe, matrix_integrator, rhs_integrator);
  test1.run(5);
}
