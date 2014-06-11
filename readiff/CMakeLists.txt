
FILE(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

FOREACH(ccfile ${sources})
  GET_FILENAME_COMPONENT(file ${ccfile} NAME_WE)
  ADD_EXECUTABLE(${file} ${ccfile} parameters.h)
  DEAL_II_SETUP_TARGET(${file})
  TARGET_LINK_LIBRARIES(${file} amandus)
ENDFOREACH()

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/rd.prm
  ${CMAKE_CURRENT_BINARY_DIR}/rd.prm COPYONLY)

ADD_CUSTOM_COMMAND(
  OUTPUT parameters.h
  COMMAND perl ${CMAKE_SOURCE_DIR}/scripts/make_parameters.pl
  ${CMAKE_CURRENT_SOURCE_DIR}/parameters.input > parameters.h
  DEPENDS parameters.input ${CMAKE_SOURCE_DIR}/scripts/make_parameters.pl)
  