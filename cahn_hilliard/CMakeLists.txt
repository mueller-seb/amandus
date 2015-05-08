ADD_EXECUTABLE(cahn_hilliard cahn_hilliard.cc)
DEAL_II_SETUP_TARGET(cahn_hilliard)
TARGET_LINK_LIBRARIES(cahn_hilliard amandus)
CONFIGURE_FILE(cahn_hilliard.prm
  ${CMAKE_CURRENT_BINARY_DIR}/cahn_hilliard.prm)

ADD_EXECUTABLE(cahn_hilliard_polynomial polynomial.cc)
DEAL_II_SETUP_TARGET(cahn_hilliard_polynomial)
TARGET_LINK_LIBRARIES(cahn_hilliard_polynomial amandus)
CONFIGURE_FILE(polynomial.prm
  ${CMAKE_CURRENT_BINARY_DIR}/cahn_hilliard_polynomial.prm)


SET(prefix cahn_hilliard)
OPTION(TEST_CAHN_HILLIARD "Test cahn hilliard." ON)
IF(TEST_CAHN_HILLIARD)
  SET(FE_ORDER 3)
  SET(INITIAL_REFINEMENT 1)

  SET(DIM 2)
  SET(LOOPS 2)
  SET(MG "false")
  FOREACH(MG "true")
    SET(TESTNAME "polynomial_mg-${MG}_dim-${DIM}")
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/polynomial.prm.cmake_template
      ${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME}.prm)
    AMANDUS_ADD_TEST(${prefix}_${TESTNAME}
      ${prefix}_polynomial ${TESTNAME}.prm) 
  ENDFOREACH(MG)

  SET(DIM 3)
  SET(LOOPS 1)
  FOREACH(MG "true")
    SET(TESTNAME "polynomial_mg-${MG}_dim-${DIM}")
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/polynomial.prm.cmake_template
      ${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME}.prm)
    AMANDUS_ADD_TEST(${prefix}_${TESTNAME}
      ${prefix}_polynomial ${TESTNAME}.prm) 
  ENDFOREACH(MG)
ENDIF()
