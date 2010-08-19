/***************************************************************************
                                nvd_device.h
                             -------------------
                               W. Michael Brown

  Utilities for dealing with cuda devices

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Thu Jan 21 2010
    copyright            : (C) 2010 by W. Michael Brown
    email                : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef NVD_DEVICE
#define NVD_DEVICE

#include <string>
#include <vector>
#include <iostream>
#include "nvd_macros.h"
#include "ucl_types.h"

namespace ucl_cudadr {

// --------------------------------------------------------------------------
// - COMMAND QUEUE STUFF
// --------------------------------------------------------------------------
typedef CUstream command_queue; 

inline void ucl_sync(CUstream &stream) {
  CU_SAFE_CALL(cuStreamSynchronize(stream));
}

struct NVDProperties {
  std::string name;
  int major;
  int minor;
  unsigned totalGlobalMem;
  int multiProcessorCount;
  CUdevprop_st p;
};

/// Class for looking at device properties
/** \note Calls to change the device outside of the class results in incorrect
  *       behavior 
  * \note There is no error checking for indexing past the number of devices **/
class UCL_Device {
 public:
  /// Collect properties for every GPU on the node
  /** \note You must set the active GPU with set() before using the device **/
  UCL_Device();
  
  ~UCL_Device();

  /// Returns 1 (For compatibility with OpenCL)
  inline int num_platforms() { return 1; }

  /// Return a string with name and info of the current platform
  std::string platform_name() { return "NVIDIA Corporation NVIDIA CUDA Driver"; }

  /// Return the number of devices that support CUDA
  inline int num_devices() { return _properties.size(); }

  /// Set the CUDA device to the specified device number
  /** A context and default command queue will be created for the device **/
  void set(int num);

  /// Get the current device number
  inline int device_num() { return _device; }

  /// Returns the default stream for the current device
  inline command_queue & cq() { return cq(0); }
  
  /// Returns the stream indexed by i
  inline command_queue & cq(const int i) { return _cq[i]; }
  
  /// Block until all commands in the default stream have completed
  inline void sync() { sync(0); }
  
  /// Block until all commands in the specified stream have completed
  inline void sync(const int i) { ucl_sync(cq(i)); }
  
  /// Get the number of command queues currently available on device
  inline int num_queues() 
    { return _cq.size(); }
  
  /// Add a stream for device computations
  inline void push_command_queue() {
    _cq.push_back(CUstream()); 
    CU_SAFE_CALL(cuStreamCreate(&_cq.back(),0)); 
  }

  /// Remove a stream for device computations
  /** \note You cannot delete the default stream **/
  inline void pop_command_queue() {
    if (_cq.size()<2) return;
    CU_SAFE_CALL(cuStreamDestroy(_cq.back()));
    _cq.pop_back();
  }
  
  /// Get the current CUDA device name
  inline std::string name() { return name(_device); }
  /// Get the CUDA device name
  inline std::string name(const int i) 
    { return std::string(_properties[i].name); }

  /// Get a string telling the type of the current device
  inline std::string device_type_name() { return device_type_name(_device); }
  /// Get a string telling the type of the device
  inline std::string device_type_name(const int i) { return "GPU"; }

  /// Get current device type (UCL_CPU, UCL_GPU, UCL_ACCELERATOR, UCL_DEFAULT)
  inline int device_type() { return device_type(_device); }
  /// Get device type (UCL_CPU, UCL_GPU, UCL_ACCELERATOR, UCL_DEFAULT)
  inline int device_type(const int i) { return UCL_GPU; }
  
  /// Returns true if double precision is support for the current device
  bool double_precision() { return double_precision(_device); }
  /// Returns true if double precision is support for the device
  bool double_precision(const int i) {return compute_cap(i)>=1.3;}
  
  /// Get the number of cores in the current device
  inline unsigned cores() { return cores(_device); }
  /// Get the number of cores
  inline unsigned cores(const int i) 
    { return _properties[i].multiProcessorCount*8; }
  
  /// Get the gigabytes of global memory in the current device
  inline double gigabytes() { return gigabytes(_device); }
  /// Get the gigabytes of global memory
  inline double gigabytes(const int i) 
    { return static_cast<double>(_properties[i].totalGlobalMem)/1073741824; }
  
  /// Get the bytes of global memory in the current device
  inline size_t bytes() { return bytes(_device); }
  /// Get the bytes of global memory
  inline size_t bytes(const int i) { return _properties[i].totalGlobalMem; }

  /// Return the GPGPU compute capability for current device
  inline double compute_cap() { return compute_cap(_device); }
  /// Return the GPGPU compute capability
  inline double compute_cap(const int i) 
    { return static_cast<double>(_properties[i].minor)/10+_properties[i].major;}
  
  /// Clock rate in GHz for current device
  inline double clock_rate() { return clock_rate(_device); }
  /// Clock rate in GHz
  inline double clock_rate(const int i) 
    { return _properties[i].p.clockRate*1e-6;}
               
  /// Get the maximum number of threads per block
  inline size_t group_size() { return group_size(_device); }
  /// Get the maximum number of threads per block
  inline size_t group_size(const int i) 
    { return _properties[i].p.maxThreadsPerBlock; }
  
  /// List all devices along with all properties
  void print_all(std::ostream &out);
 
 private:
  int _device, _num_devices;
  std::vector<NVDProperties> _properties;
  std::vector<CUstream> _cq;
  CUdevice _cu_device;
  CUcontext _context;
};

// Grabs the properties for all devices
inline UCL_Device::UCL_Device() {
  CU_SAFE_CALL(cuInit(0));
  CU_SAFE_CALL(cuDeviceGetCount(&_num_devices));
  for (int dev=0; dev<_num_devices; ++dev) {
    CUdevice m;
    CU_SAFE_CALL(cuDeviceGet(&m,dev));
    _properties.push_back(NVDProperties());
    
    char namecstr[1024];
    CU_SAFE_CALL(cuDeviceGetName(namecstr,1024,m));
    _properties.back().name=namecstr;
    
    int minor,major;
    CU_SAFE_CALL(cuDeviceComputeCapability(&_properties.back().major,
                                           &_properties.back().minor,m));
    
    CU_SAFE_CALL(cuDeviceTotalMem(&_properties.back().totalGlobalMem,m));
    CU_SAFE_CALL(cuDeviceGetAttribute(&_properties.back().multiProcessorCount,
                                      CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                      m));
    
    CU_SAFE_CALL(cuDeviceGetProperties(&_properties.back().p,m));
  }
  _device=-1;
  _cq.push_back(CUstream());
  _cq.back()=0;
}

inline UCL_Device::~UCL_Device() {
  if (_device>-1) {
    for (int i=1; i<num_queues(); i++) pop_command_queue();
    CU_SAFE_CALL(cuCtxDestroy(_context));
  }
}

// Set the CUDA device to the specified device number
inline void UCL_Device::set(int num) {
  if (_device==num)
    return;
  if (_device>-1) {
    CU_SAFE_CALL(cuCtxDestroy(_context));
    for (int i=1; i<num_queues(); i++) pop_command_queue();
  }
  CU_SAFE_CALL(cuDeviceGet(&_cu_device,num));
  CU_SAFE_CALL(cuCtxCreate(&_context,0,_cu_device));
  _device=num;
}

// List all devices along with all properties
inline void UCL_Device::print_all(std::ostream &out) {
  if (num_devices() == 0)
    out << "There is no device supporting CUDA\n";
  for (int i=0; i<num_devices(); ++i) {
    out << "\nDevice " << i << ": \"" << name(i) << "\"\n";
    out << "  Type of device:                                "
        << device_type_name(i).c_str() << std::endl;
    out << "  Compute capability:                            "
        << compute_cap(i) << std::endl;
    out << "  Double precision support:                      ";
    if (double_precision(i))
      out << "Yes\n";
    else
      out << "No\n";
    out << "  Total amount of global memory:                 "
        << gigabytes(i) << " GB\n";
    #if CUDART_VERSION >= 2000
    out << "  Number of compute units/multiprocessors:       "
        << _properties[i].multiProcessorCount << std::endl;
    out << "  Number of cores:                               "
        << cores(i) << std::endl;
    #endif
    out << "  Total amount of constant memory:               "
        << _properties[i].p.totalConstantMemory << " bytes\n";
    out << "  Total amount of local/shared memory per block: "
        << _properties[i].p.sharedMemPerBlock << " bytes\n";
    out << "  Total number of registers available per block: "
        << _properties[i].p.regsPerBlock << std::endl;
    out << "  Warp size:                                     "
        << _properties[i].p.SIMDWidth << std::endl;
    out << "  Maximum number of threads per block:           "
        << _properties[i].p.maxThreadsPerBlock << std::endl;
    out << "  Maximum group size (# of threads per block)    "
        << _properties[i].p.maxThreadsDim[0] << " x "
        << _properties[i].p.maxThreadsDim[1] << " x "
        << _properties[i].p.maxThreadsDim[2] << std::endl;
    out << "  Maximum item sizes (# threads for each dim)    "
        << _properties[i].p.maxGridSize[0] << " x "
        << _properties[i].p.maxGridSize[1] << " x "
        << _properties[i].p.maxGridSize[2] << std::endl;
    out << "  Maximum memory pitch:                          "
        << _properties[i].p.memPitch << " bytes\n";
    out << "  Texture alignment:                             "
        << _properties[i].p.textureAlign << " bytes\n";
    out << "  Clock rate:                                    "
        << clock_rate(i) << " GHz\n";
  }
}

}

#endif
