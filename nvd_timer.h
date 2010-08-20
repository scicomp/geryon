/***************************************************************************
                                 nvd_timer.h
                             -------------------
                               W. Michael Brown

  Class for timing CUDA Driver routines

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Fri Jan 22 2010
    copyright            : (C) 2010 by W. Michael Brown
    email                : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2010) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef NVD_TIMER_H
#define NVD_TIMER_H

#include "nvd_macros.h"

namespace ucl_cudadr {

/// Class for timing CUDA Driver events
class UCL_Timer {
 public:
  UCL_Timer() : _total_time(0.0f), initialized(false) { }
  UCL_Timer(UCL_Device &dev) : _total_time(0.0f), initialized(false)
    { init(dev); }
  
  ~UCL_Timer() {
    if (initialized) 
      { cuEventDestroy(start_event); cuEventDestroy(stop_event); }
  }

  /// Initialize default command queue for timing
  inline void init(UCL_Device &dev) { init(dev, dev.cq()); }
  
  /// Initialize command queue for timing
  inline void init(UCL_Device &dev, command_queue &cq) {
    _cq=cq;
    if (!initialized) {
      initialized=true;
      CU_SAFE_CALL( cuEventCreate(&start_event,0) );
      CU_SAFE_CALL( cuEventCreate(&stop_event,0) );
    }
  }
  
  /// Start timing on command queue
  inline void start() { cuEventRecord(start_event,_cq); }
  
  /// Stop timing on command queue
  inline void stop() { cuEventRecord(stop_event,_cq); }
  
  /// Set the time elapsed to zero (not the total_time)
  inline void zero() 
    { cuEventRecord(start_event,_cq); cuEventRecord(stop_event,_cq); } 
  
  /// Add time from previous start and stop to total
  /** Forces synchronization **/
  inline double add_to_total() 
    { double t=time(); _total_time+=t; return t/1000.0; }
  
  /// Return the time (ms) of last start to stop - Forces synchronization
  inline double time() { 
    float timer;
    cuEventSynchronize(stop_event);
    CU_SAFE_CALL( cuEventElapsedTime(&timer,start_event,stop_event) );
    return timer; 
  }
  
  /// Return the time (s) of last start to stop - Forces synchronization
  inline double seconds() { return time()/1000.0; }
  
  /// Return the total time in ms
  inline double total_time() { return _total_time; }

  /// Return the total time in seconds
  inline double total_seconds() { return _total_time/1000.0; }

 private:
  CUevent start_event, stop_event;
  CUstream _cq;
  double _total_time;
  bool initialized;
};

} // namespace

#endif
