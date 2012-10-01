#ifdef UCL_CUDADR
#include "nvd_device.h"
#include "nvd_timer.h"
#include "nvd_mat.h"
#include "nvd_kernel.h"
#include "smooth3d7_kernel_bin.h"
using namespace ucl_cudadr;
#endif

#ifdef UCL_OPENCL
#include "ocl_device.h"
#include "ocl_mat.h"
#include "ocl_timer.h"
#include "ocl_kernel.h"
#include "smooth3d7_kernel_str.h"
using namespace ucl_opencl;
#endif

#ifdef UCL_CUDART
#include "nvc_device.h"
#include "nvc_mat.h"
#include "nvc_timer.h"
#include "nvc_kernel.h"
#define kernel_string NULL
using namespace ucl_cudart;
#endif

#include <iostream>
#include <cstdlib>

int main(int argc, char** argv)
{
   // Set the active device
   UCL_Device dev; 
   if (dev.num_devices()==0) exit(1); 
   dev.set(0);

   // Initialize timers "segmentation fault ?"
   // UCL_Timer timer_com(dev), timer_kernel(dev);

   // Load/compile the kernel
   UCL_Program program(dev, kernel_string);
   UCL_Kernel smoothKernel(program, "smooth3d7"); 

   

   return EXIT_SUCCESS;
}

