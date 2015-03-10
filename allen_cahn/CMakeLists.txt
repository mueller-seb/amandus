
FILE(GLOB sources *.cc)
FILE(GLOB prms RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/" *.prm)
GET_FILENAME_COMPONENT(prefix ${CMAKE_CURRENT_SOURCE_DIR} NAME)

FOREACH(ccfile ${sources})
  GET_FILENAME_COMPONENT(file ${ccfile} NAME_WE)
  SET(target ${prefix}_${file})
  ADD_EXECUTABLE(${target} ${ccfile} parameters.h)
  SET_TARGET_PROPERTIES(${target} PROPERTIES OUTPUT_NAME ${file})
  DEAL_II_SETUP_TARGET(${target})
  TARGET_LINK_LIBRARIES(${target} amandus)
ENDFOREACH()

FOREACH(file ${prms})
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${file}
    ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
ENDFOREACH()

ADD_CUSTOM_COMMAND(
  OUTPUT parameters.h
  COMMAND perl ${CMAKE_SOURCE_DIR}/scripts/make_parameters.pl
  ${CMAKE_CURRENT_SOURCE_DIR}/parameters.input > parameters.h
  DEPENDS parameters.input ${CMAKE_SOURCE_DIR}/scripts/make_parameters.pl)

OPTION(TEST_ALLEN_CAHN "Test allen cahn." ON)
IF(TEST_ALLEN_CAHN)
  SET(FE_ORDER 2)
  SET(INITIAL_REFINEMENT 0)

  SET(DIM 2)
  SET(LOOPS 2)
  SET(MG "false")
  FOREACH(MG "false" "true")
    SET(TESTNAME "polynomial_mg-${MG}_dim-${DIM}")
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/polynomial.prm.cmake_template
      ${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME}.prm)
    AMANDUS_ADD_TEST(${prefix}_${TESTNAME}
      ${prefix}_polynomial ${TESTNAME}.prm) 
  ENDFOREACH(MG)

  SET(DIM 3)
  SET(LOOPS 1)
  FOREACH(MG "false" "true")
    SET(TESTNAME "polynomial_mg-${MG}_dim-${DIM}")
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/polynomial.prm.cmake_template
      ${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME}.prm)
    AMANDUS_ADD_TEST(${prefix}_${TESTNAME}
      ${prefix}_polynomial ${TESTNAME}.prm) 
  ENDFOREACH(MG)
ENDIF()
