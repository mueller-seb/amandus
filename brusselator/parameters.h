/**********************************************************************
 *  Copyright (C) 2011 - 2014 by the authors
 *  Distributed under the MIT License
 *
 * See the files AUTHORS and LICENSE in the project root directory
 **********************************************************************/
#ifndef __brusselator_parameters_h
#define __brusselator_parameters_h

#include <deal.II/base/subscriptor.h>

namespace Brusselator
{
  struct Parameters : public dealii::Subscriptor
  {
      double A;
      double B;
      double alpha0;
      double alpha1;
  };
}

#endif
