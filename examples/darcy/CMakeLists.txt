CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

ADD_EXECUTABLE(darcy_example darcy.cc)
DEAL_II_SETUP_TARGET(darcy_example)
TARGET_LINK_LIBRARIES(darcy_example amandus)
