# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT tree/cudahist package
# @author Jolly Chen, CERN
############################################################################

if (cuda)
  ROOT_LINKER_LIBRARY(InvariantMassCUDA src/InvariantMassCUDA.cu TYPE SHARED DEPENDENCIES GenVector)
  target_compile_options(InvariantMassCUDA PRIVATE -g -G -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(InvariantMassCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  ROOT_LINKER_LIBRARY(CUDAHist src/RHnCUDA.cu TYPE SHARED DEPENDENCIES MathCore)
  target_compile_options(CUDAHist PRIVATE -g -G -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(CUDAHist PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if (sycl)
  ROOT_LINKER_LIBRARY(InvariantMassSYCL src/InvariantMassSYCL.cxx TYPE SHARED DEPENDENCIES GenVector)
  add_sycl_to_target(TARGET InvariantMassSYCL SOURCES src/InvariantMassSYCL.cxx)
  target_compile_definitions(InvariantMassSYCL PUBLIC CL_TARGET_OPENCL_VERSION=300)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
