/***************************************************************************
                                nvc_kernel.h
                             -------------------
                               W. Michael Brown

  Utilities for dealing with CUDA Runtime kernels

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Fri Apr 27 2012
    copyright            : (C) 2012 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2010) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef NVC_KERNEL
#define NVC_KERNEL

#include "nvc_device.h"
#include <fstream>

namespace ucl_cudart {

class UCL_Texture;

#define UCL_MAX_KERNEL_ARGS 256
    
/// Class storing 1 or more kernel functions from a single string or file
class UCL_Program {
 public:
  inline UCL_Program(UCL_Device &device) { _cq=device.cq(); }
  inline ~UCL_Program() {}

  /// Initialize the program with a device
  inline void init(UCL_Device &device) { _cq=device.cq(); }

  /// Clear any data associated with program
  /** \note Must call init() after each clear **/
  inline void clear() { }

  /// For compatibility with CUDA Driver and OpenCL routines; Does nothing
  inline int load(const char *filename, const char *flags="",
                  std::string *log=NULL) {
    return UCL_SUCCESS;
  }
  
  /// For compatibility with CUDA Driver and OpenCL routines; Does nothing
  inline int load_string(const void *program, const char *flags="",
                         std::string *log=NULL) {
    return UCL_SUCCESS;
  }                                      
                              
  /// For compatibility with CUDA Driver and OpenCL routines; Does nothing
  inline int load_binary(const char *filename) {
    return UCL_SUCCESS;
  }
   
  friend class UCL_Kernel;
 private:
  cudaStream_t _cq;
  friend class UCL_Texture;
};

/// Class for dealing with CUDA Runtime kernels
class UCL_Kernel {
 public:
  UCL_Kernel() : _dimensions(1), _num_args(0) { 
    _param_size=0;
    _num_blocks.x=0; 
  }
  
  UCL_Kernel(UCL_Program &program, const char *function) : 
    _dimensions(1), _num_args(0) {
    _param_size=0;
    _num_blocks.x=0; 
    set_function(program,function); 
    _cq=program._cq; 
  }
  
  ~UCL_Kernel() {}

  /// Clear any function associated with the kernel
  inline void clear() { }

  /// Set the name of the kernel to be used
  /** \ret UCL_ERROR_FLAG (UCL_SUCCESS, UCL_FILE_NOT_FOUND, UCL_ERROR) **/
  inline int set_function(UCL_Program &program, const char *function) {
    _kernel=function;
    _cq=program._cq;
    return UCL_SUCCESS;
  }

  /// Set the kernel argument.
  /** If not a device pointer, this must be repeated each time the argument
    * changes 
    * \note To set kernel parameter i (i>0), parameter i-1 must be set **/
  template <class dtype>
  inline void set_arg(const unsigned index, dtype *arg) {
    if (index==_num_args)
      add_arg(arg);
    else if (index<_num_args)
      CUDA_SAFE_CALL(cudaSetupArgument((void*)arg,sizeof(dtype),
                     _offsets[index]));
    else
      assert(0==1); // Must add kernel parameters in sequential order 
  }
/* 
  /// Add a kernel argument.
  inline void add_arg(const CUdeviceptr* const arg) {
    void* ptr = (void*)(size_t)(*arg);
    _param_size = (_param_size + __alignof(ptr) - 1) & ~(__alignof(ptr) - 1);
    CUDA_SAFE_CALL(cudaSetupArgument(&ptr,sizeof(ptr),_param_size));
    _offsets.push_back(_param_size);
    _param_size+=sizeof(ptr);
    _num_args++;
    if (_num_args>UCL_MAX_KERNEL_ARGS) assert(0==1);
  }
*/
  /// Add a kernel argument.
  template <class dtype>
  inline void add_arg(dtype* arg) {
    _param_size = (_param_size+__alignof(dtype)-1) & ~(__alignof(dtype)-1);
    CUDA_SAFE_CALL(cudaSetupArgument((void*)arg,sizeof(dtype),_param_size));
    _offsets.push_back(_param_size);
    _param_size+=sizeof(dtype);
    _num_args++;
    if (_num_args>UCL_MAX_KERNEL_ARGS) assert(0==1);
  }

  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default command queue is used for the kernel execution **/
  inline void set_size(const size_t num_blocks, const size_t block_size) { 
    _dimensions=1;
    _num_blocks.x=num_blocks; 
    _num_blocks.y=1;
    _num_blocks.z=1;
    _block_size.x=block_size;
    _block_size.y=1;
    _block_size.z=1;
    CUDA_SAFE_CALL(cudaConfigureCall(_num_blocks,_block_size,0,_cq));
  }

  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default command queue for the kernel is changed to cq **/
  inline void set_size(const size_t num_blocks, const size_t block_size,
                       command_queue &cq)
    { _cq=cq; set_size(num_blocks,block_size); }

  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default command queue is used for the kernel execution **/
  inline void set_size(const size_t num_blocks_x, const size_t num_blocks_y,
                       const size_t block_size_x, const size_t block_size_y) { 
    _dimensions=2; 
    _num_blocks.x=num_blocks_x; 
    _num_blocks.y=num_blocks_y; 
    _num_blocks.z=1;
    _block_size.x=block_size_x;
    _block_size.y=block_size_y;
    _block_size.z=1;
    CUDA_SAFE_CALL(cudaConfigureCall(_num_blocks,_block_size,0,_cq));
  }
  
  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default command queue for the kernel is changed to cq **/
  inline void set_size(const size_t num_blocks_x, const size_t num_blocks_y,
                       const size_t block_size_x, const size_t block_size_y,
                       command_queue &cq) 
    {_cq=cq; set_size(num_blocks_x, num_blocks_y, block_size_x, block_size_y);}

  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default command queue is used for the kernel execution **/
  inline void set_size(const size_t num_blocks_x, const size_t num_blocks_y,
                       const size_t block_size_x, 
                       const size_t block_size_y, const size_t block_size_z) {
    _dimensions=2; 
    _num_blocks.x=num_blocks_x; 
    _num_blocks.y=num_blocks_y; 
    _num_blocks.z=1; 
    _block_size.x=block_size_x;
    _block_size.y=block_size_y;
    _block_size.z=block_size_z;
    CUDA_SAFE_CALL(cudaConfigureCall(_num_blocks,_block_size,0,_cq));
  }

  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default command queue is used for the kernel execution **/
  inline void set_size(const size_t num_blocks_x, const size_t num_blocks_y,
                       const size_t block_size_x, const size_t block_size_y,
                       const size_t block_size_z, command_queue &cq) {
    _cq=cq;
    set_size(num_blocks_x, num_blocks_y, block_size_x, block_size_y, 
             block_size_z);
  }
  
  /// Run the kernel in the default command queue
  inline void run() {
    CUDA_SAFE_CALL(cudaLaunch(_kernel.c_str()));
  }
  
  /// Clear any arguments associated with the kernel
  inline void clear_args() { 
    _num_args=0; 
    _offsets.clear(); 
    _param_size=0;
  }

  #include "ucl_arg_kludge.h"

 private:
  cudaStream_t _cq;
  std::string _kernel;
  unsigned _dimensions;
  dim3 _num_blocks;
  dim3 _block_size;
  unsigned _num_args;
  friend class UCL_Texture;
  
  std::vector<unsigned> _offsets;
  unsigned _param_size;
};

} // namespace

#endif

