/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#include <deal.II/base/data_out_base.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <amandus.h>

using namespace dealii;

AmandusParameters::AmandusParameters ()
{
  enter_subsection("Discretization");
  declare_entry("FE", "FE_Nothing", Patterns::Anything());
  declare_entry("Refinement", "1", Patterns::Integer());
  leave_subsection();
  
  enter_subsection("Newton");
  Algorithms::Newton<Vector<double> >::declare_parameters(*this);
  leave_subsection();
  
  enter_subsection("ThetaTimestepping");
  Algorithms::ThetaTimestepping<Vector<double> >::declare_parameters(*this);
  leave_subsection();
  
  enter_subsection("Output");
  DataOutInterface<2>::declare_parameters(*this);
  leave_subsection();
}


void
AmandusParameters::read(int argc, const char** argv)
{
  read_input("options.prm", true);
  std::string myname = argv[0];
  myname += ".prm";
  read_input(myname, true);
  if (argc > 1)
    read_input(argv[1], false, true);
}
