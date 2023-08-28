/*
 * Project: RooFit
 * Authors:
 *   Emmanouil Michalainas, CERN 3 March 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/**
\file Batches.h
\class Batch
\class Batches
\ingroup RooBatchCompute

These classes encapsulate the necessary data for the computations.
They are lightweight objects designed to be passed by value and also flexible,
so that they can contain data for every kind of compute function.
**/
#ifndef ROOFIT_BATCHCOMPUTE_BATCHES_H
#define ROOFIT_BATCHCOMPUTE_BATCHES_H

#include <RooBatchComputeTypes.h>

#include <cstdint>

namespace RooBatchCompute {

#ifdef __CUDACC__
// In the CPU case we use std::vector instead of fixed size arrays to pass
// around data, so no maximum size variables are necessary.
constexpr std::size_t maxParams = 8;
constexpr std::size_t maxExtraArgs = 16;
#endif // #ifdef __CUDACC__
constexpr std::size_t bufferSize = 64;

namespace RF_ARCH {

class Batch {
private:
   double _scalar = 0;
   const double *__restrict _array = nullptr;
   bool _isVector = false;

public:
   Batch() = default;
   inline Batch(InputArr array, bool isVector) : _scalar{array[0]}, _array{array}, _isVector{isVector} {}

   __roodevice__ constexpr bool isItVector() const { return _isVector; }
   inline void set(double scalar, InputArr array, bool isVector)
   {
      _scalar = scalar;
      _array = array;
      _isVector = isVector;
   }
   inline void advance(std::size_t _nEvents) { _array += _isVector * _nEvents; }
#ifdef __CUDACC__
   __roodevice__ constexpr double operator[](std::size_t i) const noexcept { return _isVector ? _array[i] : _scalar; }
#else
   constexpr double operator[](std::size_t i) const noexcept { return _array[i]; }
#endif // #ifdef __CUDACC__
};     // end class Batch

/////////////////////////////////////////////////////////////////////////////////////////////////////////

class Batches {
public:
#ifdef __CUDACC__
   // In the GPU case, we used fixed-size buffers to pass around input arrays by value.
   Batch _arrays[maxParams];
   double _extraArgs[maxExtraArgs];
#else
   Batch *_arrays;
   double *_extraArgs = nullptr;
#endif // #ifdef __CUDACC__
   std::size_t _nEvents = 0;
   std::size_t _nBatches = 0;
   std::size_t _nExtraArgs = 0;
   RestrictArr _output = nullptr;

   __roodevice__ std::size_t getNEvents() const { return _nEvents; }
   __roodevice__ std::size_t getNExtraArgs() const { return _nExtraArgs; }
   __roodevice__ double extraArg(std::size_t i) const { return _extraArgs[i]; }
   __roodevice__ void setExtraArg(std::size_t i, double val) { _extraArgs[i] = val; }
   __roodevice__ Batch operator[](int batchIdx) const { return _arrays[batchIdx]; }
   inline void setNEvents(std::size_t n) { _nEvents = n; }
   inline void advance(std::size_t nEvents)
   {
      for (std::size_t i = 0; i < _nBatches; i++)
         _arrays[i].advance(nEvents);
      _output += nEvents;
   }
}; // end class Batches

// Defines the actual argument type of the compute funciton.
using BatchesHandle = Batches &;

} // End namespace RF_ARCH
} // end namespace RooBatchCompute

#endif // #ifdef ROOFIT_BATCHCOMPUTE_BATCHES_H
