/***************************************************************************
                                 nvc_timer.h
                             -------------------
                               W. Michael Brown

  Class for timing CUDA-RT routines

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Tue Feb 3 2009
    copyright            : (C) 2009 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef NVC_TIMER_H
#define NVC_TIMER_H

#include "nvc_macros.h"
#include "nvc_device.h"

namespace ucl_cudart {

/// Class for timing CUDA events
class UCL_Timer {
 public:
  inline UCL_Timer() : _total_time(0.0f), _initialized(false) { }
  inline UCL_Timer(UCL_Device &dev) : _total_time(0.0f), _initialized(false)
    { init(dev); }

  inline ~UCL_Timer() { clear(); }

  /// Clear any data associated with timer
  /** \note init() must be called to reuse timer after a clear() **/
  inline void clear() {
    if (_initialized) { 
      CUDA_DESTRUCT_CALL(cudaEventDestroy(start_event));
      CUDA_DESTRUCT_CALL(cudaEventDestroy(stop_event));
      _initialized=false;
      _total_time=0.0;
    }
  }

  /// Initialize default command queue for timing
  inline void init(UCL_Device &dev) { init(dev, dev.cq()); }

  /// Initialize command queue for timing
  inline void init(UCL_Device &dev, command_queue &cq) {
    clear();
    _cq=cq;
    _initialized=true;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );
  }
  
  /// Start timing on command queue
  inline void start() { CUDA_SAFE_CALL(cudaEventRecord(start_event,_cq)); }
  
  /// Stop timing on command queue
  inline void stop() { CUDA_SAFE_CALL(cudaEventRecord(stop_event,_cq)); }
  
  /// Block until the start event has been reached on device
  inline void sync_start() 
    { CUDA_SAFE_CALL(cudaEventSynchronize(start_event)); }

  /// Block until the stop event has been reached on device
  inline void sync_stop() 
    { CUDA_SAFE_CALL(cudaEventSynchronize(stop_event)); }

  /// Set the time elapsed to zero (not the total_time)
  inline void zero() {
    CUDA_SAFE_CALL(cudaEventRecord(start_event,_cq));
    CUDA_SAFE_CALL(cudaEventRecord(stop_event,_cq));
  }
  
  /// Set the total time to zero
  inline void zero_total() { _total_time=0.0; }
  
  /// Add time from previous start and stop to total
  /** Forces synchronization **/
  inline double add_to_total() 
    { double t=time(); _total_time+=t; return t/1000.0; }
  
  /// Add a user specified time to the total (ms)
  inline void add_time_to_total(const double t) { _total_time+=t; }
  
  /// Return the time (ms) of last start to stop - Forces synchronization
  inline double time() { 
    float timer;
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_event));
    CUDA_SAFE_CALL( cudaEventElapsedTime(&timer,start_event,stop_event) );
    return timer; 
  }
  
  /// Return the time (s) of last start to stop - Forces synchronization
  inline double seconds() { return time()/1000.0; }
  
  /// Return the total time in ms
  inline double total_time() { return _total_time; }

  /// Return the total time in seconds
  inline double total_seconds() { return _total_time/1000.0; }

 private:
  cudaEvent_t start_event, stop_event;
  cudaStream_t _cq;
  double _total_time;
  bool _initialized;
};

} // namespace

#endif
