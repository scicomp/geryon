/***************************************************************************
                                ocl_device.h
                             -------------------
                               W. Michael Brown

  Utilities for dealing with OpenCL devices

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Mon Dec 23 2009
    copyright            : (C) 2009 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef OCL_DEVICE
#define OCL_DEVICE

#include <string>
#include <vector>
#include <iostream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl.h>
#include <CL/cl_platform.h>
#endif

#include "ocl_macros.h"
#include "ucl_types.h"

namespace ucl_opencl {
    
// --------------------------------------------------------------------------
// - COMMAND QUEUE STUFF
// --------------------------------------------------------------------------
typedef cl_command_queue command_queue; 
typedef cl_context context_type;
  
inline void ucl_sync(cl_command_queue &cq) {
  CL_SAFE_CALL(clFinish(cq));
}

struct OCLProperties {
  std::string name;
  cl_device_type device_type;
  cl_ulong global_mem;
  cl_ulong shared_mem;
  cl_ulong const_mem;
  cl_uint compute_units;
  cl_uint clock;
  size_t work_group_size;
  size_t work_item_size[3];
  bool double_precision;
  int alignment;
  size_t timer_resolution;
};

/// Class for looking at data parallel device properties
/** \note Calls to change the device outside of the class results in incorrect
  *       behavior 
  * \note There is no error checking for indexing past the number of devices **/
class UCL_Device {
 public:
  /// Collect properties for every device on the node
   /** \note You must set the active GPU with set() before using the device **/
  UCL_Device();

  /// Initialize the device with a given id
  UCL_Device(unsigned int id);
  
  ~UCL_Device();

  /// Return the number of platforms (0 if error or no platforms)
  inline int num_platforms() { return _num_platforms; }
  
  /// Return a string with name and info of the current platform
  std::string platform_name();

  /// Return the number of devices that support OpenCL
  inline int num_devices() { return _num_devices; }

  /// Set the OpenCL device to the specified device number
  /** A context and default command queue will be created for the device *
    * Returns UCL_SUCCESS if successful or UCL_ERROR if the device could not
    * be allocated for use **/
  int set(int num);

  /// Get the current device number
  inline int device_num() { return _device; }
  
  /// Returns the context for the current device
  inline cl_context & context() { return _context; }
  
  /// Returns the default stream for the current device
  inline command_queue & cq() { return cq(_default_cq); }
  
  /// Returns the stream indexed by i
  inline command_queue & cq(const int i) { return _cq[i]; }
  
  /// Set the default command queue
  /** \param i index of the command queue (as added by push_command_queue()) 
      If i is 0, the command queue created with device initialization is
      used **/
  inline void set_command_queue(const int i) { _default_cq=i; }
  
  /// Block until all commands in the default stream have completed
  inline void sync() { sync(_default_cq); }
  
  /// Block until all commands in the specified stream have completed
  inline void sync(const int i) { ucl_sync(cq(i)); }
  
  /// Get the number of command queues currently available on device
  inline int num_queues() 
    { return _cq.size(); }
  
  /// Add a command queue for device computations (with profiling enabled)
  inline void push_command_queue() {
    cl_int errorv;
    _cq.push_back(cl_command_queue());
    _cq.back()=clCreateCommandQueue(_context,_cl_device,
                                    CL_QUEUE_PROFILING_ENABLE,&errorv);
    if (errorv!=CL_SUCCESS) {
      std::cerr << "Could not create command queue on device: " << name() 
                << std::endl;
      UCL_GERYON_EXIT;
    }
  }

  /// Remove a stream for device computations
  /** \note You cannot delete the default stream **/
  inline void pop_command_queue() {
    if (_cq.size()<2) return;
    CL_SAFE_CALL(clReleaseCommandQueue(_cq.back()));
    _cq.pop_back();
  }

  /// Get the current OpenCL device name
  inline std::string name() { return name(_device); }
  /// Get the OpenCL device name
  inline std::string name(const int i) 
    { return std::string(_properties[i].name); }

  /// Get a string telling the type of the current device
  inline std::string device_type_name() { return device_type_name(_device); }
  /// Get a string telling the type of the device
  inline std::string device_type_name(const int i);
  
  /// Get current device type (UCL_CPU, UCL_GPU, UCL_ACCELERATOR, UCL_DEFAULT)
  inline int device_type() { return device_type(_device); }
  /// Get device type (UCL_CPU, UCL_GPU, UCL_ACCELERATOR, UCL_DEFAULT)
  inline int device_type(const int i);
  
  /// Returns true if host memory is efficiently addressable from device
  inline bool shared_memory() { return shared_memory(_device); }
  /// Returns true if host memory is efficiently addressable from device
  inline bool shared_memory(const int i) { return device_type(i)==UCL_CPU; }
  
  /// Returns true if double precision is support for the current device
  bool double_precision() { return double_precision(_device); }
  /// Returns true if double precision is support for the device
  bool double_precision(const int i) {return _properties[i].double_precision;}
   
  /// Get the number of cores in the current device
  inline unsigned cores() { return cores(_device); }
  /// Get the number of cores
  inline unsigned cores(const int i) 
    { if (device_type(i)==UCL_CPU) return _properties[i].compute_units;
      else return _properties[i].compute_units*8; }
  
  /// Get the gigabytes of global memory in the current device
  inline double gigabytes() { return gigabytes(_device); }
  /// Get the gigabytes of global memory
  inline double gigabytes(const int i) 
    { return static_cast<double>(_properties[i].global_mem)/1073741824; }

  /// Get the bytes of global memory in the current device
  inline size_t bytes() { return bytes(_device); }
  /// Get the bytes of global memory
  inline size_t bytes(const int i) { return _properties[i].global_mem; }
  
  /// Return the GPGPU revision number for current device
  //inline double revision() { return revision(_device); }
  /// Return the GPGPU revision number
  //inline double revision(const int i) 
  //  { return //static_cast<double>(_properties[i].minor)/10+_properties[i].major;}
  
  /// Clock rate in GHz for current device
  inline double clock_rate() { return clock_rate(_device); }
  /// Clock rate in GHz
  inline double clock_rate(const int i) { return _properties[i].clock*1e-3;}
  
  /// Return the address alignment in bytes
  inline int alignment() { return alignment(_device); }
  /// Return the address alignment in bytes
  inline int alignment(const int i) { return _properties[i].alignment; }
               
  /// Return the timer resolution
  inline size_t timer_resolution() { return timer_resolution(_device); }
  /// Return the timer resolution
  inline size_t timer_resolution(const int i) 
    { return _properties[i].timer_resolution; }
    
  /// Get the maximum number of threads per block
  inline size_t group_size() { return group_size(_device); }
  /// Get the maximum number of threads per block
  inline size_t group_size(const int i) 
    { return _properties[i].work_group_size; }
  
  /// Return the maximum memory pitch in bytes for current device
  inline size_t max_pitch() { return max_pitch(_device); }
  /// Return the maximum memory pitch in bytes
  inline size_t max_pitch(const int i) { return 0; }

  /// Returns false if accelerator cannot be shared by multiple processes
  /** If it cannot be determined, true is returned **/
  inline bool sharing_supported() { return sharing_supported(_device); }
  /// Returns false if accelerator cannot be shared by multiple processes
  /** If it cannot be determined, true is returned **/
  inline bool sharing_supported(const int i)
    { return true; }

  /// List all devices along with all properties
  void print_all(std::ostream &out);
  
  /// Return the OpenCL type for the device
  inline cl_device_id & cl_device() { return _cl_device; }
 
 private:
  int _num_platforms;          // Number of platforms
  int _platform;               // UCL_Device ID for current platform
  cl_platform_id _cl_platform; // OpenCL ID for current platform
  cl_context _context;         // Context used for accessing the device
  std::vector<cl_command_queue> _cq;// The default command queue for this device
  int _device;                            // UCL_Device ID for current device
  cl_device_id _cl_device;                // OpenCL ID for current device
  std::vector<cl_device_id> _cl_devices;  // OpenCL IDs for all devices
  int _num_devices;                       // Number of devices
  std::vector<OCLProperties> _properties; // Properties for each device
  
  void add_properties(cl_device_id);
  int create_context();
  int _default_cq;
};

// Grabs the properties for all devices
inline UCL_Device::UCL_Device() {
  cl_int errorv;
  cl_uint nplatforms;
  
  _cl_device=0;
  _device=-1;
  _num_devices=0;
  _platform=0;
  _default_cq=0;

  // --- Get Number of Platforms
  errorv=clGetPlatformIDs(1,&_cl_platform,&nplatforms);
  
  if (errorv!=CL_SUCCESS) {
    _num_platforms=0;
    return;
  } else
    _num_platforms=static_cast<int>(nplatforms);
 
  
  // --- Get Number of Devices
  cl_uint n;
  errorv=clGetDeviceIDs(_cl_platform,CL_DEVICE_TYPE_ALL,0,NULL,&n);
  _num_devices=n;
  if (errorv!=CL_SUCCESS || _num_devices==0) {
    _num_devices=0;
    return;
  }
  cl_device_id device_list[_num_devices];
  CL_SAFE_CALL(clGetDeviceIDs(_cl_platform,CL_DEVICE_TYPE_ALL,n,device_list,
                              &n));
  
  // --- Store properties for each device
  for (int i=0; i<_num_devices; i++) {
    _cl_devices.push_back(device_list[i]);
    add_properties(device_list[i]);
  }
}

// Grabs the properties for all devices
inline UCL_Device::UCL_Device(unsigned int id) {
  cl_int errorv;
  cl_uint nplatforms;
  
  _cl_device=0;
  _device=-1;
  _num_devices=0;
  _platform=0;
  _default_cq=0;

  // --- Get Number of Platforms
  errorv=clGetPlatformIDs(1,&_cl_platform,&nplatforms);
  
  if (errorv!=CL_SUCCESS) {
    _num_platforms=0;
    return;
  } else
    _num_platforms=static_cast<int>(nplatforms);
 
  
  // --- Get Number of Devices
  cl_uint n;
  errorv=clGetDeviceIDs(_cl_platform,CL_DEVICE_TYPE_ALL,0,NULL,&n);
  _num_devices=n;
  if (errorv!=CL_SUCCESS || _num_devices==0) {
    _num_devices=0;
    return;
  }
  cl_device_id device_list[_num_devices];
  CL_SAFE_CALL(clGetDeviceIDs(_cl_platform,CL_DEVICE_TYPE_ALL,n,device_list,
                              &n));
  
  // --- Store properties for each device
  for (int i=0; i<_num_devices; i++) {
    _cl_devices.push_back(device_list[i]);
    add_properties(device_list[i]);
  }
  
  set(id);
}

inline UCL_Device::~UCL_Device() {
  if (_device>-1) {
    for (size_t i=0; i<_cq.size(); i++) {
      CL_DESTRUCT_CALL(clReleaseCommandQueue(_cq.back()));
      _cq.pop_back();
    }
    CL_DESTRUCT_CALL(clReleaseContext(_context));
  }
}

inline int UCL_Device::create_context() {
  cl_int errorv;
  // MRR: Unused property at this time
  //cl_context_properties props[3];
  //props[0]=CL_CONTEXT_PLATFORM;
  //props[1]=_platform;
  //props[2]=0;
  _context=clCreateContext(0,1,&_cl_device,NULL,NULL,&errorv);
  if (errorv!=CL_SUCCESS) {
    #ifndef UCL_NO_EXIT
    std::cerr << "UCL Error: Could not access accelerator number " << _device
              << " for use.\n";
    UCL_GERYON_EXIT;
    #endif
    return UCL_ERROR;
  }
  push_command_queue();
  _default_cq=0;
  return UCL_SUCCESS;
}

inline void UCL_Device::add_properties(cl_device_id device_list) {
  OCLProperties op;
  char buffer[1024];
    
  CL_SAFE_CALL(clGetDeviceInfo(device_list,CL_DEVICE_NAME,1024,buffer,NULL));
  op.name=buffer;
  CL_SAFE_CALL(clGetDeviceInfo(device_list,CL_DEVICE_GLOBAL_MEM_SIZE,
                               sizeof(op.global_mem),&op.global_mem,NULL));
  CL_SAFE_CALL(clGetDeviceInfo(device_list,CL_DEVICE_LOCAL_MEM_SIZE,
                               sizeof(op.shared_mem),&op.shared_mem,NULL));
  CL_SAFE_CALL(clGetDeviceInfo(device_list,CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                               sizeof(op.const_mem),&op.const_mem,NULL));
  CL_SAFE_CALL(clGetDeviceInfo(device_list,CL_DEVICE_TYPE,
                               sizeof(op.device_type),&op.device_type,NULL));
  CL_SAFE_CALL(clGetDeviceInfo(device_list,CL_DEVICE_MAX_COMPUTE_UNITS,
                               sizeof(op.compute_units),&op.compute_units,
                               NULL));
  CL_SAFE_CALL(clGetDeviceInfo(device_list,CL_DEVICE_MAX_CLOCK_FREQUENCY,
                               sizeof(op.clock),&op.clock,NULL));
  CL_SAFE_CALL(clGetDeviceInfo(device_list,CL_DEVICE_MAX_WORK_GROUP_SIZE,
                               sizeof(op.work_group_size),&op.work_group_size,
                               NULL));
  CL_SAFE_CALL(clGetDeviceInfo(device_list,CL_DEVICE_MAX_WORK_ITEM_SIZES,
                               3*sizeof(op.work_item_size[0]),op.work_item_size,
                               NULL));
  CL_SAFE_CALL(clGetDeviceInfo(device_list,CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                               sizeof(cl_uint),&op.alignment,NULL));
  op.alignment/=8;                               
  
  // Determine if double precision is supported
  cl_uint double_width;
  CL_SAFE_CALL(clGetDeviceInfo(device_list,
                               CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                               sizeof(double_width),&double_width,NULL));
  if (double_width==0)
    op.double_precision=false;
  else
    op.double_precision=true;
  
  CL_SAFE_CALL(clGetDeviceInfo(device_list,
                               CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                               sizeof(size_t),&op.timer_resolution,NULL));
  
  _properties.push_back(op);
}

inline std::string UCL_Device::platform_name() {
  char info[1024];
  
  CL_SAFE_CALL(clGetPlatformInfo(_cl_platform,CL_PLATFORM_VENDOR,1024,info,
                                 NULL));
  std::string ans=std::string(info)+' ';
  
  CL_SAFE_CALL(clGetPlatformInfo(_cl_platform,CL_PLATFORM_NAME,1024,info,
                                 NULL));
  ans+=std::string(info)+' ';
  
  CL_SAFE_CALL(clGetPlatformInfo(_cl_platform,CL_PLATFORM_VERSION,1024,info,
               NULL));
  ans+=std::string(info);
  
  return ans;
}

// Get a string telling the type of the device
inline std::string UCL_Device::device_type_name(const int i) {
  if (_properties[i].device_type==CL_DEVICE_TYPE_CPU)
    return "CPU";
  else if (_properties[i].device_type==CL_DEVICE_TYPE_GPU)
    return "GPU";
  else if (_properties[i].device_type==CL_DEVICE_TYPE_ACCELERATOR)
    return "ACCELERATOR";
  else
    return "DEFAULT";
}

// Get a string telling the type of the device
inline int UCL_Device::device_type(const int i) {
  if (_properties[i].device_type==CL_DEVICE_TYPE_CPU)
    return UCL_CPU;
  else if (_properties[i].device_type==CL_DEVICE_TYPE_GPU)
    return UCL_GPU;
  else if (_properties[i].device_type==CL_DEVICE_TYPE_ACCELERATOR)
    return UCL_ACCELERATOR;
  else
    return UCL_DEFAULT;
}

// Set the CUDA device to the specified device number
inline int UCL_Device::set(int num) {
  if (_device==num)
    return UCL_SUCCESS;
  
  if (_device>-1) {
    for (size_t i=0; i<_cq.size(); i++) {
      CL_SAFE_CALL(clReleaseCommandQueue(_cq.back()));
      _cq.pop_back();
    }
    CL_SAFE_CALL(clReleaseContext(_context));
  }
  
  cl_device_id device_list[_num_devices];
  cl_uint n;
  CL_SAFE_CALL(clGetDeviceIDs(_cl_platform,CL_DEVICE_TYPE_ALL,_num_devices,
                               device_list,&n));

  _device=num;
  _cl_device=device_list[_device];
  return create_context();
}

// List all devices along with all properties
inline void UCL_Device::print_all(std::ostream &out) {
  if (num_devices() == 0)
    out << "There is no device supporting OpenCL\n";
  for (int i=0; i<num_devices(); ++i) {
    out << "\nDevice " << i << ": \"" << name(i).c_str() << "\"\n";
    out << "  Type of device:                                "
        << device_type_name(i).c_str() << std::endl;
    out << "  Double precision support:                      ";
    if (double_precision(i))
      out << "Yes\n";
    else
      out << "No\n";
    out << "  Total amount of global memory:                 "
        << gigabytes(i) << " GB\n";
    out << "  Number of compute units/multiprocessors:       "
        << _properties[i].compute_units << std::endl;
    //out << "  Number of cores:                               "
    //    << cores(i) << std::endl;
    out << "  Total amount of constant memory:               "
        << _properties[i].const_mem << " bytes\n";
    out << "  Total amount of local/shared memory per block: "
        << _properties[i].shared_mem << " bytes\n";
    //out << "  Total number of registers available per block: "
    //    << _properties[i].regsPerBlock << std::endl;
    //out << "  Warp size:                                     "
    //    << _properties[i].warpSize << std::endl;
    out << "  Maximum group size (# of threads per block)    "
        << _properties[i].work_group_size << std::endl;
    out << "  Maximum item sizes (# threads for each dim)    "
        << _properties[i].work_item_size[0] << " x "
        << _properties[i].work_item_size[1] << " x "
        << _properties[i].work_item_size[2] << std::endl;
    //out << "  Maximum sizes of each dimension of a grid:     "
    //    << _properties[i].maxGridSize[0] << " x " 
    //    << _properties[i].maxGridSize[1] << " x "
    //    << _properties[i].maxGridSize[2] << std::endl;
    //out << "  Maximum memory pitch:                          "
    //    << _properties[i].memPitch) << " bytes\n";
    //out << "  Texture alignment:                             "
    //    << _properties[i].textureAlignment << " bytes\n";
    out << "  Clock rate:                                    "
        << clock_rate(i) << " GHz\n";
    //out << "  Concurrent copy and execution:                 ";
  }
}

}

#endif
