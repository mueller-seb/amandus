/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2010, 2012, 2013
 *
 **********************************************************************/

#include <deal.II/base/data_out_base.h>
#include <amandus.h>

using namespace dealii;

AmandusParameters::AmandusParameters ()
{
  enter_subsection("Discretization");
  declare_entry("FE", "", Patterns::Anything());
  declare_entry("Refinement", "1", Patterns::Integer());
  leave_subsection();
  
  enter_subsection("Output");
  DataOutInterface<2>::declare_parameters(*this);
leave_subsection();
}


void
AmandusParameters::read(int argc, const char** argv)
{
read_input("options", true);
//read_input(argv[0], true);
if (argc > 1)
  read_input(argv[1], false, true);
}
