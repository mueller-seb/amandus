/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
/**
 * @file
 * <ul>
 * <li> Maxwell operator</li>
 * <li> Dirichlet boundary condition</li>
 * <li> Eigenvalue problem</li>
 * <li> UMFPack</li>
 * <li> <a
 * href="http://perso.univ-rennes1.fr/monique.dauge/benchmax.html">Benchmark
 * problem</a> published by Monique Dauge
 * </ul>
 *
 * @ingroup Examples
 */

#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/dof_output_operator.h>
#include <deal.II/numerics/dof_output_operator.templates.h>
#include <apps.h>
#include <amandus_arpack.h>
#include <maxwell/eigen.h>

#include <boost/scoped_ptr.hpp>

int main(int argc, const char** argv)
{
  const unsigned int d=2;
  
  std::ofstream logfile("deallog");
  deallog.attach(logfile);
  
  AmandusParameters param;
  param.declare_entry("Eigenvalues", "12", Patterns::Integer());
  param.read(argc, argv);
  param.log_parameters(deallog);
  
  param.enter_subsection("Discretization");
  boost::scoped_ptr<const FiniteElement<d> > fe(FETools::get_fe_from_name<d>(param.get("FE")));
  
  Triangulation<d> tr;
  GridGenerator::hyper_L (tr, -1, 1);
  tr.refine_global(param.get_integer("Refinement"));
  param.leave_subsection();
  
  MaxwellIntegrators::DivCurl::Eigen<d> matrix_integrator;
  AmandusUMFPACK<d> app(tr, *fe);
  app.parse_parameters(param);
  app.set_boundary(0);
  app.set_number_of_matrices(2);
  AmandusArpack<d> solver(app, matrix_integrator);
  app.control.set_reduction(1.e-10);
  
  global_refinement_eigenvalue_loop(param.get_integer("Steps"),
				    param.get_integer("Eigenvalues"),
				    app, solver);
}










