CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
FIND_PACKAGE(deal.II 8.0 REQUIRED
  HINTS
  ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(cell-matrices)

ADD_LIBRARY(amandus
  amandus.cc)

DEAL_II_SETUP_TARGET(amandus)  

FOREACH(file cc_01 st_01 st_01n da_01n la_01n ac_01 residual_laplace_01 residual_darcy_01)
  ADD_EXECUTABLE(${file} ${file}.cc)
  DEAL_II_SETUP_TARGET(${file})
  TARGET_LINK_LIBRARIES(${file} amandus)
ENDFOREACH()
