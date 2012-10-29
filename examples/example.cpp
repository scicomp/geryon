/***************************************************************************
                                 example.cpp

  Vector add example (Host Code)
 
 ***************************************************************************/

#include "geryon.h"
 
#ifdef UCL_CUDADR
#include "example_kernel_bin.h"
#endif

#ifdef UCL_OPENCL
#include "example_kernel_str.h"
#endif

#ifdef UCL_CUDART
#define kernel_string NULL
#endif

int main() { 
  // Set the active device
  UCL_Device dev; 
  if (dev.num_devices()==0) exit(1); 
  dev.set(0);

  // Initialize timers
  UCL_Timer timer_com(dev), timer_kernel(dev);
 
  // Load/compile the kernel
  UCL_Program program(dev,kernel_string);
  UCL_Kernel k_vec_add(program,"vec_add");

  // Allocate storage on host and device for input
  // - Vector is double precision on host and single on accelerator
  // - If dev is a CPU or shares memory with the host, no allocation
  //   is performed on the device
  UCL_Vector<double,float> a(6,dev), b(6,dev);
  // Allocate memory for answer
  UCL_D_Vec<float> answer(6,dev,UCL_WRITE_ONLY);
 
  // Get the data on the host
  for (int i=0; i<6; i++) { a[i]=i; b[i]=i; } 

  // Get the data on the device (copy ignored if device is a CPU, but
  //   typecast is performed if necessary.
  timer_com.start();
  a.update_device();
  b.update_device();
  timer_com.stop();
 
  // Set up 1-dimensional kernel grid to add 6 elements and run on device
  timer_kernel.start();
  size_t num_blocks=6, block_size=1;
  k_vec_add.set_size(num_blocks,block_size);

  // Enqueue the kernel in the default command queue to be run
  k_vec_add.run(&a,&b,&answer);
  timer_kernel.stop();

  // Print the results
  std::cout << "Answer: " << answer << std::endl
            << "Input copy time: " << timer_com.seconds() << std::endl
            << "Kernel time: " << timer_kernel.seconds() << std::endl;  
 
  return 0;
}
