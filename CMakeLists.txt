CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
FIND_PACKAGE(deal.II 8.3 REQUIRED
  HINTS
  ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(amandus)
SET(amandus_VERSION 0.1.0)

# List of subdirectories with example codes
SET(EXAMPLES laplace advection advection-diffusion
  stokes maxwell
  allen_cahn cahn_hilliard
  brusselator readiff
  elasticity
  )
# To be added after fixing: darcy brinkman lotkavolterra

# List of toplevel header files
FILE(GLOB HEADERS *.h)

INCLUDE(CTest)

# keep a reference from build directory to headers in source directory
EXECUTE_PROCESS(
  COMMAND
  mkdir -p "${CMAKE_BINARY_DIR}/include")
EXECUTE_PROCESS(
  COMMAND
  ln -nsf ${PROJECT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/include/amandus)
# include headers found in src directory via symbolic link
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
# include headers generated during build with scripts/make_parameters.pl
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

FILE(COPY ${PROJECT_SOURCE_DIR}/scripts
  DESTINATION ${CMAKE_BINARY_DIR}/bin/amandus)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
ADD_LIBRARY(amandus
  amandus.cc amandus_parameters.cc amandus_sparse.cc amandus_sparse_multigrid.cc)
SET_PROPERTY(TARGET amandus PROPERTY VERSION ${amandus_VERSION})
SET_PROPERTY(TARGET amandus APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )

DEAL_II_SETUP_TARGET(amandus)

#
# Install files
#
SET(INCLUDE_INSTALL_DIR include)
SET(LIB_INSTALL_DIR lib)
SET(RUNTIME_INSTALL_DIR bin)
INSTALL(TARGETS amandus EXPORT amandusTargets
  LIBRARY DESTINATION ${LIB_INSTALL_DIR}
  ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
  RUNTIME DESTINATION ${RUNTIME_INSTALL_DIR}
  INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
)

INSTALL(DIRECTORY 
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake"
  DESTINATION ${LIB_INSTALL_DIR})

INSTALL(FILES ${HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/amandus)

INSTALL(DIRECTORY ${EXAMPLES}
  DESTINATION ${INCLUDE_INSTALL_DIR}/amandus
  FILES_MATCHING PATTERN "*.h")

INSTALL(DIRECTORY scripts
  DESTINATION ${RUNTIME_INSTALL_DIR}/amandus)


#
# Write config files
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
#

INCLUDE(CMakePackageConfigHelpers)
WRITE_BASIC_PACKAGE_VERSION_FILE(
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/amandus/amandusConfigVersion.cmake"
  VERSION ${amandus_VERSION}
  COMPATIBILITY AnyNewerVersion
)

CONFIGURE_PACKAGE_CONFIG_FILE(
  cmake/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/amandus/amandusConfig.cmake"
  INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/amandus
  PATH_VARS INCLUDE_INSTALL_DIR)

export(TARGETS amandus
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/amandus/amandusTargets.cmake")

install(EXPORT amandusTargets
  FILE amandusTargets.cmake
  DESTINATION ${LIB_INSTALL_DIR}/cmake/amandus)

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

FOREACH (example ${EXAMPLES})
  ADD_SUBDIRECTORY(${example} EXCLUDE_FROM_ALL)
ENDFOREACH(example)

ADD_SUBDIRECTORY(tests)
ADD_SUBDIRECTORY(doc)
