# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT tree/experimental package
# @author Jolly Chen, CERN
# @author Maksymilian Graczyk, CERN
############################################################################

if (cuda)
  ROOT_LINKER_LIBRARY(InvariantMassCUDA src/InvariantMassCUDA.cu TYPE SHARED DEPENDENCIES GenVector)
  target_compile_options(InvariantMassCUDA PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(InvariantMassCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(InvariantMassCUDA PUBLIC ROOT_RDF_CUDA)

  ROOT_LINKER_LIBRARY(CUDAHist src/RHnCUDA.cu TYPE SHARED DEPENDENCIES MathCore)
  target_compile_options(CUDAHist PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(CUDAHist PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(CUDAHist PUBLIC ROOT_RDF_CUDA)
endif()

if (sycl)
  find_package(OpenSYCL REQUIRED)
  ROOT_LINKER_LIBRARY(InvariantMassSYCL src/InvariantMassSYCL.cxx TYPE SHARED DEPENDENCIES GenVector)
  target_compile_definitions(InvariantMassSYCL PUBLIC ROOT_RDF_SYCL)

  if (cuda)
    target_compile_definitions(InvariantMassSYCL PUBLIC ROOT_RDF_CUDA)
  endif()

  add_sycl_to_target(TARGET InvariantMassSYCL SOURCES src/InvariantMassSYCL.cxx)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
