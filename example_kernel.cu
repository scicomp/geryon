/***************************************************************************
                              example_kernel.cpp
                             -------------------
                               W. Michael Brown

  Kernel for Vector Add Example

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

#ifdef NV_KERNEL
#define __global  
#define GLOBAL_ID_X threadIdx.x+blockIdx.x*blockDim.x
#define __kernel extern "C" __global__
#else
#define GLOBAL_ID_X get_global_id(0)
#endif

__kernel void vec_add(__global Scalar *a, __global Scalar *b, 
                      __global Scalar *ans) {
  int i=GLOBAL_ID_X;
  ans[i]=a[i]+b[i];
}

