
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

SET(DIM 2)
FOREACH(DIM 2 3)
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/polynomial.prm.cmake_template
    ${CMAKE_CURRENT_BINARY_DIR}/polynomial_${DIM}d.prm)
  AMANDUS_ADD_TEST(${prefix}_polynomial_${DIM}d_test ${prefix}_polynomial
    polynomial_${DIM}d.prm) 
ENDFOREACH(DIM)
