# ordinary cmake file to build test
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(Boost COMPONENTS unit_test_framework REQUIRED)

ADD_EXECUTABLE(foo_useage foo_useage.cc)
TARGET_LINK_LIBRARIES(foo_useage ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

ADD_EXECUTABLE(bar_useage bar_useage.cc)
TARGET_LINK_LIBRARIES(bar_useage ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
