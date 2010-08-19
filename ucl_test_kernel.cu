/***************************************************************************
                              ucl_test_kernel.cu
                             --------------------
                               W. Michael Brown

  Test code for UCL (vector add).

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Thu Feb 11 2010
    copyright            : (C) 2010 by W. Michael Brown
    email                : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2010) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifdef NV_KERNEL

#define DEV_PTR Scalar
#define GLOBAL_ID_X threadIdx.x+__mul24(blockIdx.x,blockDim.x)
#define __kernel extern "C" __global__

#else

#define DEV_PTR __global Scalar
#define GLOBAL_ID_X get_global_id(0)

#endif

__kernel void vec_add(DEV_PTR *a, DEV_PTR *b, DEV_PTR *ans) 
  {  Ordinal i=GLOBAL_ID_X;  ans[i]=a[i]+b[i];}

