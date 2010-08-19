/***************************************************************************
                                ocl_kernel.h
                             -------------------
                               W. Michael Brown

  Utilities for dealing with OpenCL kernels

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Sun Feb 7 2010
    copyright            : (C) 2010 by W. Michael Brown
    email                : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2010) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef OCL_KERNEL
#define OCL_KERNEL

#include "ocl_device.h"
#include <fstream>

namespace ucl_opencl {
    
/// Class storing 1 or more kernel functions from a single string or file
class UCL_Program {
 public:
  UCL_Program(UCL_Device &device) : _device(device.cl_device()),
                                     _context(device.context()), 
                                     _cq(device.cq()) {
    CL_SAFE_CALL(clRetainContext(_context)); 
    CL_SAFE_CALL(clRetainCommandQueue(_cq));
  }
    
  ~UCL_Program() {
     CL_SAFE_CALL(clReleaseProgram(_program)); 
     CL_SAFE_CALL(clReleaseContext(_context));
     CL_SAFE_CALL(clReleaseCommandQueue(_cq));
   }
  
  /// Load a program from a file and compile with flags
  inline int load(const char *filename, const char *flags="",
                  std::string *log=NULL) {
    std::ifstream in(filename);
    if (!in || in.is_open()==false) {
      #ifndef UCL_NO_EXIT 
      std::cerr << "UCL Error: Could not open kernel file: " 
                << filename << std::endl;
      exit(1);
      #endif
      return UCL_FILE_NOT_FOUND;
    }
  
    std::string program((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    in.close();
    return load_string(program.c_str(),flags,log);
  }
  
  /// Load a program from a string and compile with flags
  inline int load_string(const char *program, const char *flags="",
                         std::string *log=NULL) {
    cl_int error_flag;
    const char *prog=program;
    _program=clCreateProgramWithSource(_context,1,&prog,NULL,&error_flag);
    CL_CHECK_ERR(error_flag);
    error_flag = clBuildProgram(_program,1,&_device,flags,NULL,NULL);
    cl_build_status build_status;
    CL_SAFE_CALL(clGetProgramBuildInfo(_program,_device,
                                       CL_PROGRAM_BUILD_STATUS, 
                                       sizeof(cl_build_status),&build_status,
                                       NULL));
                                       
    if (build_status != CL_SUCCESS || log!=NULL) {
      size_t ms;
      CL_SAFE_CALL(clGetProgramBuildInfo(_program,_device,CL_PROGRAM_BUILD_LOG,0, 
                                         NULL, &ms));
      char build_log[ms];                                     
      CL_SAFE_CALL(clGetProgramBuildInfo(_program,_device,CL_PROGRAM_BUILD_LOG,ms,
                                         build_log, NULL));
                                         
      if (log!=NULL)
        *log=std::string(build_log);
                                                 
      if (build_status != CL_SUCCESS) {
        #ifndef UCL_NO_EXIT                                                 
        std::cerr << std::endl
                  << "----------------------------------------------------------\n"
                  << " UCL Error: Error compiling OpenCL Program...\n"
                  << "----------------------------------------------------------\n";
        std::cerr << build_log << std::endl;
        #endif
        return UCL_COMPILE_ERROR;
      }
    }
    
    return UCL_SUCCESS;
  }                                               
   
  friend class UCL_Kernel;
 private:
  cl_program _program;
  cl_device_id _device; 
  cl_context _context;
  cl_command_queue _cq;
};

/// Class for dealing with OpenCL kernels
class UCL_Kernel {
 public:
  UCL_Kernel() : _dimensions(1), _num_args(0) 
    {  _block_size[0]=0; _num_blocks[0]=0; }
  
  UCL_Kernel(UCL_Program &program, const char *function) : 
    _dimensions(1), _num_args(0)
    {  _block_size[0]=0; _num_blocks[0]=0; set_function(program,function); }

  ~UCL_Kernel();

  /// Get the kernel function from a program
  /** \return UCL_ERROR_FLAG (UCL_SUCCESS, UCL_FILE_NOT_FOUND, UCL_ERROR) **/
  inline int set_function(UCL_Program &program, const char *function);

  /// Set the kernel argument.
  /** If not a device pointer, this must be repeated each time the argument
    * changes **/
  template <class dtype>
  inline void set_arg(const cl_uint index, dtype *arg) { 
    CL_SAFE_CALL(clSetKernelArg(_kernel,index,sizeof(dtype),arg)); 
    if (index>_num_args) _num_args=index;
  }
 
  /// Add a kernel argument.
  template <class dtype>
  inline void add_arg(dtype *arg) {
    CL_SAFE_CALL(clSetKernelArg(_kernel,_num_args,sizeof(dtype),arg)); 
    _num_args++; 
  }

  /// Set the number of thread blocks and the number of threads in each block
  inline void set_size(const size_t num_blocks, const size_t block_size) { 
    _dimensions=1; 
    _num_blocks[0]=num_blocks*block_size; 
    _block_size[0]=block_size; 
  }

  /// Set the number of thread blocks and the number of threads in each block
  inline void set_size(const size_t num_blocks_x, const size_t num_blocks_y,
                       const size_t block_size_x, const size_t block_size_y) { 
    _dimensions=2; 
    _num_blocks[0]=num_blocks_x*block_size_x; 
    _block_size[0]=block_size_x; 
    _num_blocks[1]=num_blocks_y*block_size_y; 
    _block_size[1]=block_size_y; 
  }
  
  /// Set the number of thread blocks and the number of threads in each block
  inline void set_size(const size_t num_blocks_x, const size_t num_blocks_y,
                       const size_t block_size_x, 
                       const size_t block_size_y, const size_t block_size_z) {
    _dimensions=3; 
    const size_t num_blocks_z=1;
    _num_blocks[0]=num_blocks_x*block_size_x; 
    _block_size[0]=block_size_x; 
    _num_blocks[1]=num_blocks_y*block_size_y; 
    _block_size[1]=block_size_y; 
    _num_blocks[2]=num_blocks_z*block_size_z; 
    _block_size[2]=block_size_z; 
  }

  /// Run the kernel in the default command queue
  inline void run() {
    run(_cq);
  }
  
  /// Run the kernel in the specified command queue
  inline void run(command_queue &cq) {
    CL_SAFE_CALL(clEnqueueNDRangeKernel(cq,_kernel,_dimensions,NULL,
                                        _num_blocks,_block_size,0,NULL,NULL));
  }
  
  /// Clear any arguments associated with the kernel
  inline void clear_args() { _num_args=0; }

  #include "ucl_arg_kludge.h"
  
 private:
  cl_kernel _kernel;
  cl_program _program;
  cl_uint _dimensions;
  size_t _block_size[3];
  size_t _num_blocks[3];
  
  cl_command_queue _cq;        // The default command queue for this kernel
  unsigned _num_args;
};

inline UCL_Kernel::~UCL_Kernel() {
//  clReleaseKernel(_kernel);
  clReleaseProgram(_program); 
  clReleaseCommandQueue(_cq);
}

inline int UCL_Kernel::set_function(UCL_Program &program, const char *function) {
  _cq=program._cq;
  CL_SAFE_CALL(clRetainCommandQueue(_cq));
  _program=program._program;
  CL_SAFE_CALL(clRetainProgram(_program));
  cl_int error_flag;
  _kernel=clCreateKernel(program._program,function,&error_flag);
  
  if (error_flag!=CL_SUCCESS) {
    #ifndef UCL_NO_EXIT
    std::cerr << "UCL Error: Could not find function: " << function
              << " in program.\n";
    exit(1);
    #endif
    return UCL_FUNCTION_NOT_FOUND;
  }
  return UCL_SUCCESS;                                               
}

} // namespace

#endif

