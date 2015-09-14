CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

# sources
INCLUDE_DIRECTORIES(include)
FILE(GLOB sources source/*.cc)

# Change here the name of executable as you want, replace laplace
ADD_EXECUTABLE(laplace ${sources} parameters.h)

# Amandus headers and static library location
IF(DEFINED ENV{AMANDUS})
  INCLUDE_DIRECTORIES($ENV{AMANDUS}/source)
  ADD_LIBRARY(amandus STATIC IMPORTED)
  SET_TARGET_PROPERTIES(amandus PROPERTIES
     IMPORTED_LOCATION $ENV{AMANDUS}/build/libamandus.a)

# again change target laplace to as you want
  TARGET_LINK_LIBRARIES(laplace amandus)
ELSE()
 MESSAGE(FATAL_ERROR "SET ENV_VARIABLE FOR AMANDUS")
ENDIF()

## Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3")        ## Optimize
    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()

# DEAL.II package configuration
FIND_PACKAGE(deal.II 8.0 REQUIRED
  HINTS

# provide the path for dealii install directory
  ${DEAL_II_DIR} /scratch/dealii/install/ ../../ $ENV{DEAL_II_DIR})

# change laplace to your target file
DEAL_II_SETUP_TARGET(laplace)

ADD_CUSTOM_COMMAND(
  OUTPUT parameters.h
  COMMAND perl $ENV{AMANDUS}/source/scripts/make_parameters.pl
  ${CMAKE_CURRENT_SOURCE_DIR}/parameters.input > ${CMAKE_CURRENT_SOURCE_DIR}/include/parameters.h

# again change this path to as above
  DEPENDS parameters.input $ENV{AMANDUS}/source/scripts/make_parameters.pl)
