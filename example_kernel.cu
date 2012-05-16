// **********************************************************************
//                              example_kernel.cu
//                             -------------------
//                               W. Michael Brown
//
//  Kernel for Vector Add Example
//
//
// **********************************************************************

#ifdef NV_KERNEL
#define __global  
#define GLOBAL_ID_X threadIdx.x+blockIdx.x*blockDim.x
#define __kernel extern "C" __global__
#else
#define GLOBAL_ID_X get_global_id(0)
#endif

#define Scalar float

__kernel void vec_add(__global Scalar *a, __global Scalar *b, 
                      __global Scalar *ans) {
  int i=GLOBAL_ID_X;
  ans[i]=a[i]+b[i];
}

