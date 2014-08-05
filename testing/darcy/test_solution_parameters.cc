#define BOOST_TEST_MODULE test_solution_parameters
#include <boost/test/included/unit_test.hpp>

#include <darcy/solution_parameters.h>

#define TOL 0.001
BOOST_AUTO_TEST_CASE(test_parameters)
{
  std::vector<double> coefficient_parameters;
  coefficient_parameters.push_back(100.0);
  coefficient_parameters.push_back(1.0);
  coefficient_parameters.push_back(100.0);
  coefficient_parameters.push_back(1.0);
  
  std::vector<double> control_solution_parameters;
  control_solution_parameters.push_back(0.126902);
  control_solution_parameters.push_back(-11.5926);
  control_solution_parameters.push_back(0.785398);

  DarcyCoefficient::SolutionParameters
    solution_parameters(coefficient_parameters);

  BOOST_CHECK_EQUAL(solution_parameters.parameters.size(),
                    control_solution_parameters.size());
  for(unsigned int i = 0; i < solution_parameters.parameters.size(); ++i)
  {
    BOOST_CHECK_CLOSE(solution_parameters.parameters[i],
                      control_solution_parameters[i], TOL);
  }
}
