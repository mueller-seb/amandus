CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
FIND_PACKAGE(deal.II 8.0 REQUIRED
  HINTS
  ${DEAL_II_DIR} ~/svn-dealii/install ../ ../../ $ENV{DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(cell-matrices)

ADD_LIBRARY(amandus
  amandus.cc amandus_sparse.cc amandus_sparse_multigrid.cc)

DEAL_II_SETUP_TARGET(amandus)  

ADD_SUBDIRECTORY(tests)

