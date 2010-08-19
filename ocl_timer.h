/***************************************************************************
                                 ocl_timer.h
                             -------------------
                               W. Michael Brown

  Class for timing OpenCL routines

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Jan Fri 22 2010
    copyright            : (C) 2010 by W. Michael Brown
    email                : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2010) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef OCL_TIMER_H
#define OCL_TIMER_H

#include "ocl_macros.h"

namespace ucl_opencl {

/// Class for timing OpenCL events
class UCL_Timer {
 public:
  UCL_Timer() : _total_time(0.0f), initialized(false) { }
  UCL_Timer(UCL_Device &dev) : _total_time(0.0f), initialized(false)
    { init(dev); }
  
  ~UCL_Timer() 
    { if (initialized) { CL_SAFE_CALL(clReleaseCommandQueue(_cq)); } }

  /// Initialize default command queue for timing
  inline void init(UCL_Device &dev) { init(dev,dev.cq()); }
  
  /// Initialize command queue for timing
  inline void init(UCL_Device &dev, command_queue &cq) {
    t_factor=dev.timer_resolution()/1000000000.0;   
    if (initialized)
      CL_SAFE_CALL(clReleaseCommandQueue(_cq));
    _cq=cq;
    clRetainCommandQueue(_cq);
    CL_SAFE_CALL(clSetCommandQueueProperty(_cq,CL_QUEUE_PROFILING_ENABLE,
                                           CL_TRUE,NULL));
    initialized=true;
  }
  
  /// Start timing on default command queue
  inline void start() { clEnqueueMarker(_cq,&start_event); }
  
  /// Stop timing on default command queue
  inline void stop() { clEnqueueMarker(_cq,&stop_event); }
  
  /// Set the time elapsed to zero (not the total_time)
  inline void zero() 
    { clEnqueueMarker(_cq,&start_event); clEnqueueMarker(_cq,&stop_event); } 
  
  /// Add time from previous start and stop to total
  /** Forces synchronization **/
  inline void add_to_total() { _total_time+=time(); }
  
  /// Return the time (ms) of last start to stop - Forces synchronization
  inline double time() {
    cl_ulong tstart,tend;
    CL_SAFE_CALL(clWaitForEvents(1,&stop_event));
    CL_SAFE_CALL(clGetEventProfilingInfo(stop_event,
                                         CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &tend, NULL));
    CL_SAFE_CALL(clGetEventProfilingInfo(start_event,
                                         CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), &tstart, NULL));
    return (tend-tstart)*t_factor; 
  }
  
  /// Return the time (s) of last start to stop - Forces synchronization
  inline double seconds() { return time()/1000.0; }
  
  /// Return the total time in ms
  inline double total_time() { return _total_time; }

  /// Return the total time in seconds
  inline double total_seconds() { return _total_time/1000.0; }

 private:
  cl_event start_event, stop_event;
  cl_command_queue _cq;
  double _total_time;
  bool initialized;
  double t_factor;
};

} // namespace

#endif
