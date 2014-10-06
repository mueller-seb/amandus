#ifndef __advectiondiffusion_parameters_h
#define __advectiondiffusion_parameters_h

#include <deal.II/base/parameter_handler.h>

namespace AdvectionDiffusion
{
struct Parameters : public dealii::Subscriptor
{
    static void declare_parameters(dealii::ParameterHandler& param);
    void parse_parameters(dealii::ParameterHandler& param);
};

inline
void
Parameters::declare_parameters(dealii::ParameterHandler& param)
{
}

inline
void
Parameters::parse_parameters(dealii::ParameterHandler& param)
{
}

}
#endif
