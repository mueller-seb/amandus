# CMake generated Testfile for 
# Source directory: /home/anja/deal/dealii/amandus/testing
# Build directory: /home/anja/deal/dealii/amandus/testing
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
ADD_TEST(foo_useage "/home/anja/deal/dealii/amandus/testing/useage/foo_useage")
ADD_TEST(bar_useage "/home/anja/deal/dealii/amandus/testing/useage/bar_useage")
ADD_TEST(darcy_test "/home/anja/deal/dealii/amandus/testing/darcy/darcy_test")
SUBDIRS(useage)
SUBDIRS(darcy)
