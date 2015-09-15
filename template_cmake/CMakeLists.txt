CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

# sources
INCLUDE_DIRECTORIES(include)
FILE(GLOB sources source/*.cc)
FILE(GLOB prms RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/" *.prm)
GET_FILENAME_COMPONENT(prefix ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Amandus headers and static library location
IF(DEFINED ENV{AMANDUS})
  INCLUDE_DIRECTORIES($ENV{AMANDUS}/source)
  ADD_LIBRARY(amandus STATIC IMPORTED)
  SET_TARGET_PROPERTIES(amandus PROPERTIES
     IMPORTED_LOCATION $ENV{AMANDUS}/build/libamandus.a)
# DEAL.II package configuration
  FIND_PACKAGE(deal.II 8.0 REQUIRED
  HINTS
  ${DEAL_II_DIR} /scratch/dealii/install/ ../../ $ENV{DEAL_II_DIR})
ELSE()
   MESSAGE(FATAL_ERROR "SET ENV_VARIABLE FOR AMANDUS")
ENDIF()

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

## Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3")        ## Optimize
    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()

ADD_CUSTOM_COMMAND(
  OUTPUT parameters.h
  COMMAND perl $ENV{AMANDUS}/source/scripts/make_parameters.pl
  ${CMAKE_CURRENT_SOURCE_DIR}/parameters.input > ${CMAKE_CURRENT_SOURCE_DIR}/include/parameters.h

# again change this path to as above
  DEPENDS parameters.input $ENV{AMANDUS}/source/scripts/make_parameters.pl)

