// $Id: cc_01.cc 1396 2014-02-02 19:37:44Z kanschat $

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include "apps.h"
#include "matrix_curl_curl.h"
#include "rhs_one.h"


int main()
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  Triangulation<d> tr;
  GridGenerator::hyper_L (tr, -1, 1);

  const unsigned int degree = 1;
  FE_Nedelec<d> vec(degree);
  FE_Q<d> scal(degree);
  FESystem<d> fe(vec, 1, scal, 1);
  
  CurlCurlMatrix<d> matrix_integrator;
  RhsOne<d> rhs_integrator;
  
  AmandusApplication<d> app(tr, fe, matrix_integrator, rhs_integrator);
  AmandusResidual<d> residual(app, rhs_integrator);
  app.control.set_reduction(1.e-10);
  
  global_refinement_linear_loop(5, app, residual);
}
