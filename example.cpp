/***************************************************************************
                                 example.cpp
                             -------------------
                               W. Michael Brown

  Vector add example

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Mon March 8 2010
    copyright            : (C) 2010 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2010) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifdef USE_CUDA_DRIVER
#include "nvd_device.h"
#include "nvd_timer.h"
#include "nvd_mat.h"
#include "nvd_kernel.h"
#define KERNEL_NAME "example_kernel.ptx"
using namespace ucl_cudadr;
#endif

#ifdef USE_OPENCL
#include "ocl_device.h"
#include "ocl_mat.h"
#include "ocl_timer.h"
#include "ocl_kernel.h"
#define KERNEL_NAME "example_kernel.cu"
using namespace ucl_opencl;
#endif

using namespace std;

int main() {  
  // Set the active device and initialize timers
  UCL_Device dev;  if (dev.num_devices()==0) exit(1);  dev.set(0);
  UCL_Timer timer_com(dev), timer_kernel(dev);
  
  // Load/compile the kernel
  UCL_Program program(dev);
  string flags="-cl-fast-relaxed-math -D Scalar=float";
  program.load(KERNEL_NAME,flags.c_str());
  UCL_Kernel k_vec_add(program,"vec_add");

  // Allocate storage on host and device
  UCL_H_Vec<double> a(6,dev,UCL_WRITE_OPTIMIZED), b(6,dev,UCL_WRITE_OPTIMIZED);
  UCL_D_Vec<float> dev_a(6,dev,UCL_READ_ONLY), dev_b(6,dev,UCL_READ_ONLY);
  UCL_D_Vec<float> answer(6,dev,UCL_WRITE_ONLY);
  
  // Get the data on the device
  for (int i=0; i<6; i++) { a[i]=i; b[i]=i; }  
  timer_com.start();
  ucl_copy(dev_a,a,true);
  ucl_copy(dev_b,b,true);
  timer_com.stop();
  
  // Set up 1-dimensional kernel grid to add 6 elements and run on device
  timer_kernel.start();
  k_vec_add.add_args(&dev_a.begin(),&dev_b.begin(),&answer.begin());
  size_t num_blocks=6, block_size=1;
  k_vec_add.set_size(num_blocks,block_size);

  // Enqueue the kernel in the default command queue
  k_vec_add.run();
  timer_kernel.stop();
  cout << "Answer: " << answer << endl 
       << "Input copy time: " << timer_com.seconds() << endl 
       << "Kernel time: " << timer_kernel.seconds() << endl;   
  
  return 0;
}

