/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#include <deal.II/base/data_out_base.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <amandus/amandus.h>

using namespace dealii;

AmandusParameters::AmandusParameters ()
{
  declare_entry("Steps", "3", Patterns::Integer());

  enter_subsection("Discretization");
  declare_entry("FE", "FE_Nothing", Patterns::Anything());
  declare_entry("Refinement", "1", Patterns::Integer());
  leave_subsection();

  enter_subsection("Linear Solver");
  ReductionControl::declare_parameters(*this);
  set("Reduction","1.e-10");
  leave_subsection();

  enter_subsection("Multigrid");
  declare_entry("Sort", "false", Patterns::Bool());
  declare_entry("Interior smoothing", "true", Patterns::Bool());
  declare_entry("Smoothing steps on leaves", "1", Patterns::Integer(0));
  declare_entry("Variable smoothing steps", "false", Patterns::Bool());
  leave_subsection();
  
  Algorithms::Newton<Vector<double> >::declare_parameters(*this);  
  Algorithms::ThetaTimestepping<Vector<double> >::declare_parameters(*this);
  
  enter_subsection("Output");
  DataOutInterface<2>::declare_parameters(*this);
  set("Output format","vtu");
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
