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
    email                : wmbrown@sandia.gov
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

namespace ucl_cudart {

/// Class for timing CUDA events
class UCL_Timer {
 public:
  UCL_Timer() : _total_time(0.0f), initialized(false) { }
  UCL_Timer(UCL_Device &dev) : _total_time(0.0f), initialized(false)
    { init(dev); }
  
  ~UCL_Timer() {
    if (initialized) 
      { cudaEventDestroy(start_event); cudaEventDestroy(stop_event); }
  }

  /// Initialize default command queue for timing
  inline void init(UCL_Device &dev) { init(dev, dev.cq()); }
  
  /// Initialize command queue for timing
  inline void init(UCL_Device &dev, command_queue &cq) {
    _cq=cq;
    if (!initialized) {
      initialized=true;
      CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
      CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );
    }
  }
  
  /// Start timing on command queue
  inline void start() { cudaEventRecord(start_event,_cq); }
  
  /// Stop timing on command queue
  inline void stop() { cudaEventRecord(stop_event,_cq); }
  
  /// Set the time elapsed to zero (not the total_time)
  inline void zero() 
    { cudaEventRecord(start_event,_cq); cudaEventRecord(stop_event,_cq); } 
  
  /// Add time from previous start and stop to total
  /** Forces synchronization **/
  inline void add_to_total() { _total_time+=time(); }
  
  /// Return the time (ms) of last start to stop - Forces synchronization
  inline double time() { 
    float timer;
    cudaEventSynchronize(stop_event);
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
  bool initialized;
};

} // namespace

#endif
