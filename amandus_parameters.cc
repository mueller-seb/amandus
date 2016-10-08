/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 *
 **********************************************************************/

#include <amandus/amandus.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/theta_timestepping.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/path_search.h>

using namespace dealii;

AmandusParameters::AmandusParameters()
{
  declare_entry("Steps", "3", Patterns::Integer());

  enter_subsection("Discretization");
  declare_entry("FE", "FE_Nothing", Patterns::Anything());
  declare_entry("Refinement", "1", Patterns::Integer());
  leave_subsection();

  enter_subsection("Linear Solver");
  ReductionControl::declare_parameters(*this);
  set("Reduction", "1.e-10");
  declare_entry("Use Right Preconditioning", "true", Patterns::Bool());
  declare_entry("Use Default Residual", "true", Patterns::Bool());
  leave_subsection();

  enter_subsection("Multigrid");
  declare_entry("Sort", "false", Patterns::Bool());
  declare_entry("Interior smoothing", "true", Patterns::Bool());
  declare_entry("Include exterior smoothing on blocks", "", Patterns::List(Patterns::Integer(0)));
  declare_entry("Smoothing steps on leaves", "1", Patterns::Integer(0));
  declare_entry("Variable smoothing steps", "false", Patterns::Bool());
  declare_entry("Smoother Relaxation", "1.0", Patterns::Double());
  declare_entry("Log Smoother Statistics", "false", Patterns::Bool());
  leave_subsection();

  Algorithms::Newton<Vector<double>>::declare_parameters(*this);
  Algorithms::ThetaTimestepping<Vector<double>>::declare_parameters(*this);

  enter_subsection("Output");
  DataOutInterface<2>::declare_parameters(*this);
  set("Output format", "vtu");
  leave_subsection();

  enter_subsection("Arpack");
  declare_entry("Min Arnoldi vectors", "20", Patterns::Integer(0));
  declare_entry("Symmetric", "false", Patterns::Bool());
  declare_entry("Max steps", "100", Patterns::Integer(1));
  declare_entry("Tolerance", "1.e-10", Patterns::Double());
  leave_subsection();
}

void
AmandusParameters::read(int argc, const char** argv)
{
  try
  {
    parse_input("options.prm");
  }
  catch (dealii::PathSearch::ExcFileNotFound&)
  {
  }

  std::string myname = argv[0];
  myname += ".prm";
  parse_input(myname);
  if (argc > 1)
  {
    try
    {
      parse_input(argv[1]);
    }
    catch (dealii::PathSearch::ExcFileNotFound&)
    {
      std::ofstream out(argv[1]);
      print_parameters(out, OutputStyle::ShortText);
    }
  }
}
