# Follow this example for adding tests. Put new tests in a subdirectory with
# its own CMakeLists.txt. Add the subdirectory here and append all tests to
# be covered to the variable coveredTests.
OPTION(TEST_USEAGE "Test useage." ON)
IF(TEST_USEAGE)
  ADD_SUBDIRECTORY(useage)
  LIST(APPEND coveredTests foo_useage)
  LIST(APPEND coveredTests bar_useage)
ENDIF()

OPTION(TEST_DARCY "Test darcy." ON)
IF(TEST_DARCY)
  ADD_SUBDIRECTORY(darcy)
  LIST(APPEND coveredTests test_darcy_integrators)
  LIST(APPEND coveredTests test_darcy_polynomial)
  LIST(APPEND coveredTests test_darcy_solution_parameters)
  LIST(APPEND coveredTests test_darcy_estimator)
  LIST(APPEND coveredTests test_darcy_checkerboard)
ENDIF()

OPTION(TEST_ADVECTION "Test advection." ON)
IF (TEST_ADVECTION)
  LIST(APPEND coveredTests advection_polynomial_01)
ENDIF()

# No need to modify behind this; should be sufficient for most cases.
# make tests available to ctest
FOREACH(test ${coveredTests})
  ADD_DEPENDENCIES(test_suite ${test})
  ADD_TEST(NAME ${test} COMMAND ${test})
ENDFOREACH(test)
