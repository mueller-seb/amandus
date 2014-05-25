// $Id$

#ifndef __brusselator_parameters_h
#define __brusselator_parameters_h

#include <deal.II/base/subscriptor.h>

namespace Brusselator
{
  struct Parameters : public dealii::Subscriptor
  {
      double A;
      double B;
      double alpha;
  };
}

#endif
