#define BOOST_TEST_MODULE useage
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE(simple_test)
{
  BOOST_CHECK_EQUAL(2 + 3, 5);
}
