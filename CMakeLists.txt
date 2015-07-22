CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
FIND_PACKAGE(deal.II 8.0 REQUIRED
  HINTS
  ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(amandus)

INCLUDE(CTest)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR})

ADD_LIBRARY(amandus
  amandus.cc amandus_parameters.cc amandus_sparse.cc amandus_sparse_multigrid.cc)

DEAL_II_SETUP_TARGET(amandus)  

# CMake's generated test target runs the added tests but does not build them,
# thus we provide custom targets which builds all tests added with the
# function AMANDUS_ADD_TEST() and runs them.
ADD_CUSTOM_TARGET(test_suite)
ADD_CUSTOM_TARGET(run_tests 
  COMMAND ${CMAKE_CTEST_COMMAND}
  DEPENDS test_suite)

FUNCTION(AMANDUS_ADD_TEST test_name test_exec)
  IF(BUILD_TESTING)
    ADD_DEPENDENCIES(test_suite ${test_exec})
    ADD_TEST(NAME ${test_name} COMMAND ${test_exec} ${ARGN})
  ENDIF()
ENDFUNCTION()

# ADD_SUBDIRECTORY(testing)

ADD_SUBDIRECTORY(elasticity)
ADD_SUBDIRECTORY(advection)
ADD_SUBDIRECTORY(allen_cahn)
ADD_SUBDIRECTORY(cahn_hilliard)
ADD_SUBDIRECTORY(brusselator)
ADD_SUBDIRECTORY(maxwell)
ADD_SUBDIRECTORY(readiff)
ADD_SUBDIRECTORY(tests)
ADD_SUBDIRECTORY(doc)
ADD_SUBDIRECTORY(laplace)
ADD_SUBDIRECTORY(stokes)
ADD_SUBDIRECTORY(advection-diffusion)
