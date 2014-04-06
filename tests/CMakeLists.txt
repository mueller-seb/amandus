
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

FOREACH(file cc_01 st_01 st_01u st_01n da_01n da_02n la_01 la_01u la_01n ac_01 residual_laplace_01 residual_darcy_01 residual_stokes_01)
  ADD_EXECUTABLE(${file} ${file}.cc)
  DEAL_II_SETUP_TARGET(${file})
  TARGET_LINK_LIBRARIES(${file} amandus)
ENDFOREACH()
