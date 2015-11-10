CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(amandus REQUIRED HINTS $ENV{AMANDUS_DIR})

# Headers, sources and parameter files
# This will create an executable for each main file present in the
# directory containing CMakeLists.txt
INCLUDE_DIRECTORIES(include)
FILE(GLOB sources source/*.cc)
FILE(GLOB mainfiles *.cc)
FILE(GLOB paramfiles *.prm)

# Loops over all executables to be compiled
# Targets and executables will have the same name and will be compiled
# separately.
# If you are not using parameters file then take out parameters.h from
# the ADD_EXECUTABLE and comment rest of the code after loop.
FOREACH(mainfile ${mainfiles})
  GET_FILENAME_COMPONENT(basename ${mainfile} NAME_WE)
  ADD_EXECUTABLE(${basename} ${mainfile} ${sources} parameters.h)
  TARGET_LINK_LIBRARIES(${basename} amandus)
  DEAL_II_SETUP_TARGET(${basename})
ENDFOREACH()

# Copies the parameter files into the building directory
FOREACH(paramfile ${paramfiles})
  GET_FILENAME_COMPONENT(filename ${paramfile} NAME)
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${filename}
    ${CMAKE_CURRENT_BINARY_DIR}/${filename} COPYONLY)
ENDFOREACH()

# Generates parameter files
ADD_CUSTOM_COMMAND(
  OUTPUT parameters.h
  COMMAND perl ${AMANDUS_SCRIPTS_DIR}/make_parameters.pl
  ${CMAKE_CURRENT_SOURCE_DIR}/parameters.input > ${CMAKE_CURRENT_SOURCE_DIR}/include/parameters.h
  DEPENDS parameters.input ${AMANDUS_SCRIPTS_DIR}/make_parameters.pl)
