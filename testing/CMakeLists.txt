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



# No need to modify behind this; should be sufficient for most cases.
# make tests available to ctest
FOREACH(test ${coveredTests})
  ADD_TEST(NAME ${test} COMMAND ${test})
ENDFOREACH(test)

# add target to build and run all tests
ADD_CUSTOM_TARGET(build_tests DEPENDS ${coveredTests})
ADD_CUSTOM_TARGET(run_tests COMMAND ${CMAKE_CTEST_COMMAND} -V DEPENDS build_tests)
