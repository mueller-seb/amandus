/**********************************************************************
 * $Id$
 *
 * Copyright Guido Kanschat, 2014
 *
 **********************************************************************/

#ifndef __brinkman_parameters_h
#define __brinkman_parameters_h

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/logstream.h>

/**
 * Classes and functions pertaining the discretization of coupled
 * Brinkman-Darcy-Stokes problems.
 *
 * The bilinear form considered is
 * @f[
 * \begin{array}{ccccl}
 *  a(u,v) &-& b(v,p) &=& (f,v) \\
 *  b(u,q) && &=& (g,q)
 * \end{array}
 * @f]
 * The operators are
 * @f{eqnarray*}{
 *  a(u,v) &=& \bigl(\nu \nabla u, \nabla v\bigr)
 *    + \bigl(\rho u,v\bigr)
 *    + \bigl<\gamma\sqrt\rho u_{S,\tau},v_{S,\tau}\bigr>_{\Gamma_{SD}}
 *    + {\rm IP} + \bigl<\sigma \nabla\!\cdot\! u, \nabla\!\cdot\! v \bigr>
 *  \\
 *  b(u,q) &=& \bigl(\nabla\!\cdot\! u,q\bigr)
 * @f}
 * Here, IP refers to the interior penalty face terms. The last term
 * in the form \f$a(.,.)\f$ is an optional grad-div stabilization.
 *
 * @ingroup integrators
 */
namespace Brinkman
{
  /**
   * The parameters common to matrix and residual computations.  The
   * vectors #viscosity, #resistance, and #graddiv_stabilization are
   * indexed by the cell material id and represent the coefficients in
   * the system.
   *
   * The coefficients refer to the equations described in the
   * namespace Brinkman.
   */
  class Parameters : public dealii::Subscriptor
  {
    public:
				     /**
				      * Default constructor, only
				      * setting the Saffman friction
				      * parameter.
				      */
    Parameters();
    
				     /**
				      * Constructor, initializing
				      * default values for #viscosity
				      * and #resistance. These
				      * refer to a situation with two
				      * subdomains, one Darcy and one
				      * Stokes. They are
				      * zero resistance and viscosity
				      * one in material 0 (Stokes).
				      * In material 1 (Darcy), the
				      * viscosity is 0 and the
				      * resistance is the function
				      * argument.
				      */
    Parameters(double resistance);
      /**
       * @brief The vector of Stokes/Brinkman viscosities \f$\mu\f$
       */
      std::vector<double> viscosity;
      /**
       * @brief The vector of Darcy/Brinkman resistances \f$\rho\f$
       */
      std::vector<double> resistance;
      /**
       * @brief The vector of coefficients \f$\sigma\f$ for grad-div stabilization
       */      
      std::vector<double> graddiv_stabilization;

      /**
       * The additional coefficient \f$\gamma\f$ of the friction term.
       */
      double saffman;

  };
  
  
  inline
  Parameters::Parameters()
		:
		  saffman(.1)
  {}
  
  
  inline
  Parameters::Parameters(double res)
		  :
		  viscosity(2, 0.),
		  resistance(2, 0.),
		  graddiv_stabilization(2, 1.),
		  saffman(.1)
  {
    dealii::deallog << "Brinkman " << res << std::endl;
    viscosity[0] = 1.;
    resistance[1] = res;
  }
}

#endif
