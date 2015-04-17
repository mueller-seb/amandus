ADD_EXECUTABLE(cahn_hilliard cahn_hilliard.cc)
DEAL_II_SETUP_TARGET(cahn_hilliard)
TARGET_LINK_LIBRARIES(cahn_hilliard amandus)
CONFIGURE_FILE(cahn_hilliard.prm
  ${CMAKE_CURRENT_BINARY_DIR}/cahn_hilliard.prm)
