/***************************************************************************
                                nvc_memory.h
                             -------------------
                               W. Michael Brown

  CUDA Specific Memory Management and Vector/Matrix Containers

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Thu Jun 25 2009
    copyright            : (C) 2009 by W. Michael Brown
    email                : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef NVC_MEMORY_H
#define NVC_MEMORY_H

#include <iostream>
#include <cassert>
#include <string.h>
#include "nvc_macros.h"
#include "ucl_types.h"

namespace ucl_cudart {

// --------------------------------------------------------------------------
// - API Specific Types
// --------------------------------------------------------------------------
typedef dim3 ucl_kernel_dim;

// --------------------------------------------------------------------------
// - HOST MEMORY ALLOCATION ROUTINES
// --------------------------------------------------------------------------
template <class mat_type, class copy_type>
inline int _host_alloc(mat_type &mat, copy_type &cm, const size_t n,  
                        const enum UCL_MEMOPT kind) {
  cudaError err;
  if (kind==UCL_RW_OPTIMIZED)  
    err=cudaMallocHost((void **)mat.host_ptr(),n);
  else if (kind==UCL_WRITE_OPTIMIZED)
    err=cudaHostAlloc((void **)mat.host_ptr(),n,cudaHostAllocWriteCombined);
  else {
    *(mat.host_ptr())=(typename mat_type::data_type*)malloc(n);
    err=cudaSuccess;
  }
  if (err!=cudaSuccess || *(mat.host_ptr())==NULL)
    return UCL_MEMORY_ERROR;
  return UCL_SUCCESS;
}

template <class mat_type>
inline int _host_alloc(mat_type &mat, UCL_Device &dev, const size_t n,  
                        const enum UCL_MEMOPT kind) {
  cudaError err;
  if (kind==UCL_RW_OPTIMIZED)  
    err=cudaMallocHost((void **)mat.host_ptr(),n);
  else if (kind==UCL_WRITE_OPTIMIZED)
    err=cudaHostAlloc((void **)mat.host_ptr(),n,cudaHostAllocWriteCombined);
  else {
    *(mat.host_ptr())=(typename mat_type::data_type*)malloc(n);
    err=cudaSuccess;
  }
  if (err!=cudaSuccess || *(mat.host_ptr())==NULL)
    return UCL_MEMORY_ERROR;
  return UCL_SUCCESS;
}

template <class mat_type>
inline void _host_free(mat_type &mat, const enum UCL_MEMOPT kind) {
  if (kind!=UCL_NOT_PINNED)
    CUDA_SAFE_CALL(cudaFreeHost(mat.begin()));
  else
    free(mat.begin());
}

// --------------------------------------------------------------------------
// - DEVICE MEMORY ALLOCATION ROUTINES
// --------------------------------------------------------------------------
template <class mat_type, class copy_type>
inline int _device_alloc(mat_type &mat, copy_type &cm, const size_t n,
                         const enum UCL_MEMOPT kind) {
  cudaError err=cudaMalloc(mat.cbegin(),n);
  if (err==cudaSuccess)
    return UCL_SUCCESS;
  return UCL_MEMORY_ERROR;
}

template <class mat_type>
inline int _device_alloc(mat_type &mat, UCL_Device &dev, const size_t n,
                         const enum UCL_MEMOPT kind) {
  cudaError err=cudaMalloc(mat.cbegin(),n);
  if (err==cudaSuccess)
    return UCL_SUCCESS;
  return UCL_MEMORY_ERROR;
}

template <class mat_type, class copy_type>
inline int _device_alloc(mat_type &mat, copy_type &cm, const size_t rows,
                          const size_t cols, size_t &pitch,
                          const enum UCL_MEMOPT kind) {
  cudaError err=cudaMallocPitch(mat.cbegin(),&pitch,
                                cols*sizeof(typename mat_type::data_type),rows);
  if (err==cudaSuccess)
    return UCL_SUCCESS;
  return UCL_MEMORY_ERROR;
}    

template <class mat_type, class copy_type>
inline int _device_alloc(mat_type &mat, UCL_Device &d, const size_t rows,
                          const size_t cols, size_t &pitch,
                          const enum UCL_MEMOPT kind) {
  cudaError err=cudaMallocPitch(mat.cbegin(),&pitch,
                                cols*sizeof(typename mat_type::data_type),rows);
  if (err==cudaSuccess)
    return UCL_SUCCESS;
  return UCL_MEMORY_ERROR;
}    

template <class mat_type>
inline void _device_free(mat_type &mat) {
  CUDA_SAFE_CALL(cudaFree(mat.begin()));
}

template <class numtyp>
inline void _device_view(numtyp **ptr, numtyp *in) {
  (*ptr)=in;
}

template <class numtyp>
inline void _device_view(numtyp **ptr, numtyp *in, const size_t offset,
                         const size_t numsize) {
  (*ptr)=in+offset;
}

// --------------------------------------------------------------------------
// - DEVICE IMAGE ALLOCATION ROUTINES
// --------------------------------------------------------------------------
template <class mat_type, class copy_type>
inline void _device_image_alloc(mat_type &mat, copy_type &cm, const size_t rows,
                                const size_t cols) {
  CUDA_SAFE_CALL(cudaMallocArray(mat.cbegin(),mat.channel(),cols,rows));
}    

template <class mat_type, class copy_type>
inline void _device_image_alloc(mat_type &mat, UCL_Device &d, const size_t rows,
                         const size_t cols) {
  CUDA_SAFE_CALL(cudaMallocArray(mat.cbegin(),mat.channel(),cols,rows));
}    

template <class mat_type>
inline void _device_image_free(mat_type &mat) {
  CUDA_SAFE_CALL(cudaFreeArray(mat.begin()));
}

// --------------------------------------------------------------------------
// - ZERO ROUTINES
// --------------------------------------------------------------------------
inline void _host_zero(void *ptr, const size_t n) {
  memset(ptr,0,n);
}

template <class mat_type>
inline void _device_zero(mat_type &mat, const size_t n) {
  CUDA_SAFE_CALL(cudaMemset(mat.begin(),0,n));
}

// --------------------------------------------------------------------------
// - TEXTURE SPECIFIC ROUTINES
// --------------------------------------------------------------------------

// Get a channel for float array
template <class numtyp>
inline void cuda_gb_get_channel(cudaChannelFormatDesc &channel) {
  channel = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
}

// Get a channel for float2 array
template <>
inline void cuda_gb_get_channel<float2>(cudaChannelFormatDesc &channel) {
  channel = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
}

// Get a channel for double array
template <>
inline void cuda_gb_get_channel<double>(cudaChannelFormatDesc &channel) {
  channel = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);
}

// Get a channel for double array
template <>
inline void cuda_gb_get_channel<double2>(cudaChannelFormatDesc &channel) {
  channel = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
}

// Get a channel for int array
template <>
inline void cuda_gb_get_channel<int>(cudaChannelFormatDesc &channel) {
  channel = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
}

// --------------------------------------------------------------------------
// - MEMCPY ROUTINES
// --------------------------------------------------------------------------

// Determine the CUDA transfer order based on matrix traits
template<int host1,int host2> struct _CUDA_TRAN;
template<> struct _CUDA_TRAN<0,0> 
{ static inline enum cudaMemcpyKind a() { return cudaMemcpyDeviceToDevice; } };
template<> struct _CUDA_TRAN<0,1> 
{ static inline enum cudaMemcpyKind a() { return cudaMemcpyHostToDevice; } };
template<> struct _CUDA_TRAN<1,0> 
{ static inline enum cudaMemcpyKind a() { return cudaMemcpyDeviceToHost; } };
template<int host1,int host2> struct _CUDA_TRAN 
{ static inline enum cudaMemcpyKind a() { return cudaMemcpyHostToHost; } };

template<int image, int image2> struct _ucl_memcpy;

// Both are textures
template<> struct _ucl_memcpy<2,2> {
  template <class p1, class p2>
  static inline void mc(p1 dst, const p2 src, const size_t n,
                        const enum cudaMemcpyKind kind) {
    CUDA_SAFE_CALL(cudaMemcpyArrayToArray(dst,0,0,src,0,0,n,kind));
  }
  template <class p1, class p2>
  static inline void mc(p1 dst, const p2 src, const size_t n,
                        const enum cudaMemcpyKind kind, cudaStream_t &cq) {
    CUDA_SAFE_CALL(cudaMemcpyArrayToArrayAsync(dst,0,0,src,0,0,n,kind,cq));
  }
  template <class p1, class p2>
  static inline void mc(p1 dst, const size_t dpitch, const p2 src, 
                        const size_t spitch, const size_t cols,
                        const size_t rows, const enum cudaMemcpyKind kind) {
    CUDA_SAFE_CALL(cudaMemcpy2DArrayToArray(dst,0,0,src,0,0,cols,rows,kind));

  }
  template <class p1, class p2>
      static inline void mc(p1 dst, const size_t dpitch, const p2 src, 
                            const size_t spitch, const size_t cols,
                            const size_t rows, const enum cudaMemcpyKind kind,
                            cudaStream_t &cq) {
    CUDA_SAFE_CALL(cudaMemcpy2DArrayToArrayAsync(dst,0,0,src,0,0,cols,rows,
                                                 kind,cq));
  }
};

// Destination is texture
template<int image2> struct _ucl_memcpy<2,image2> {
  template <class p1, class p2>
  static inline void mc(p1 dst, const p2 src, const size_t n,
                        const enum cudaMemcpyKind kind) {
    CUDA_SAFE_CALL(cudaMemcpyToArray(dst,0,0,src,n,kind));
  }
  template <class p1, class p2>
  static inline void mc(p1 dst, const p2 src, const size_t n,
                        const enum cudaMemcpyKind kind, cudaStream_t &cq) {
    CUDA_SAFE_CALL(cudaMemcpyToArrayAsync(dst,0,0,src,n,kind,cq));
  }
  template <class p1, class p2>
  static inline void mc(p1 dst, const size_t dpitch, const p2 src, 
                        const size_t spitch, const size_t cols,
                        const size_t rows, const enum cudaMemcpyKind kind) {
    CUDA_SAFE_CALL(cudaMemcpy2DToArray(dst,0,0,src,spitch,cols,rows,kind));
  }
  template <class p1, class p2>
  static inline void mc(p1 dst, const size_t dpitch, const p2 src, 
                        const size_t spitch, const size_t cols,
                        const size_t rows, const enum cudaMemcpyKind kind,
                        cudaStream_t &cq) {
    CUDA_SAFE_CALL(cudaMemcpy2DToArrayAsync(dst,0,0,src,spitch,cols,rows,
                                            kind,cq));
  }
};

// Source is texture
template<int image1> struct _ucl_memcpy<image1,2> {
  template <class p1, class p2>
  static inline void mc(p1 dst, const p2 src, const size_t n,
                        const enum cudaMemcpyKind kind) {
    CUDA_SAFE_CALL(cudaMemcpyFromArray(dst,src,0,0,n,kind));
  }
  template <class p1, class p2>
  static inline void mc(p1 dst, const p2 src, const size_t n,
                        const enum cudaMemcpyKind kind, cudaStream_t &cq) {
    CUDA_SAFE_CALL(cudaMemcpyFromArray(dst,src,0,0,n,kind,cq));
  }
  template <class p1, class p2>
      static inline void mc(p1 dst, const size_t dpitch, const p2 src, 
                            const size_t spitch, const size_t cols,
                            const size_t rows, const enum cudaMemcpyKind kind) {
    CUDA_SAFE_CALL(cudaMemcpy2DFromArray(dst,dpitch,src,0,0,cols,rows,kind));
  }
  template <class p1, class p2>
  static inline void mc(p1 dst, const size_t dpitch, const p2 src, 
                        const size_t spitch, const size_t cols,
                        const size_t rows, const enum cudaMemcpyKind kind,
                        cudaStream_t &cq) {
    CUDA_SAFE_CALL(cudaMemcpy2DFromArrayAsync(dst,dpitch,src,0,0,cols,rows,
                                              kind,cq));
  }
};

// Neither are textures
template <int image1, int image2> struct _ucl_memcpy {
  template <class p1, class p2>
  static inline void mc(p1 dst, const p2 src, const size_t n,
                        const enum cudaMemcpyKind kind) {
    CUDA_SAFE_CALL(cudaMemcpy(dst,src,n,kind));
  }
  template <class p1, class p2>
  static inline void mc(p1 dst, const p2 src, const size_t n,
                        const enum cudaMemcpyKind kind, cudaStream_t &cq) {
    CUDA_SAFE_CALL(cudaMemcpyAsync(dst,src,n,kind,cq));
  }
  template <class p1, class p2>
      static inline void mc(p1 dst, const size_t dpitch, const p2 src, 
                            const size_t spitch, const size_t cols,
                            const size_t rows, const enum cudaMemcpyKind kind) {
    CUDA_SAFE_CALL(cudaMemcpy2D(dst,dpitch,src,spitch,cols,rows,kind));
  }
  template <class p1, class p2>
  static inline void mc(p1 dst, const size_t dpitch, const p2 src, 
                        const size_t spitch, const size_t cols,
                        const size_t rows, const enum cudaMemcpyKind kind,
                        cudaStream_t &cq) {
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(dst,dpitch,src,spitch,cols,rows,kind,
                                     cq));
  }
};

template<class mat1, class mat2>
inline void ucl_mv_cpy(mat1 &dst, const mat2 &src, const size_t n) {
  _ucl_memcpy<mat1::MEM_TYPE,mat2::MEM_TYPE>::mc(dst.begin(),src.begin(),n,
                                                 _CUDA_TRAN<mat1::MEM_TYPE,
                                                        mat2::MEM_TYPE>::a());
}

template<class mat1, class mat2>
inline void ucl_mv_cpy(mat1 &dst, const mat2 &src, const size_t n,
                       cudaStream_t &cq) {
  #ifdef UCL_DEBUG
  if (mat1::MEM_TYPE==1 && mat2::MEM_TYPE==0)
    assert(dst.kind()==UCL_RW_OPTIMIZED || dst.kind()==UCL_WRITE_OPTIMIZED);
  if (mat2::MEM_TYPE==1 && mat1::MEM_TYPE==0)
    assert(src.kind()==UCL_RW_OPTIMIZED || src.kind()==UCL_WRITE_OPTIMIZED);
  #endif
  _ucl_memcpy<mat1::MEM_TYPE,mat2::MEM_TYPE>::mc(dst.begin(),src.begin(),n,
                                                 _CUDA_TRAN<mat1::MEM_TYPE,
                                                          mat2::MEM_TYPE>::a(),
                                                            cq);
}

template<class mat1, class mat2>
inline void ucl_mv_cpy(mat1 &dst, const size_t dpitch, const mat2 &src, 
                       const size_t spitch, const size_t cols, 
                       const size_t rows) {
  _ucl_memcpy<mat1::MEM_TYPE,mat2::MEM_TYPE>::mc(dst.begin(),dpitch,src.begin(),
                                                 spitch,cols,rows,
                                                 _CUDA_TRAN<mat1::MEM_TYPE,
                                                        mat2::MEM_TYPE>::a());
}

template<class mat1, class mat2>
    inline void ucl_mv_cpy(mat1 &dst, const size_t dpitch, const mat2 &src, 
                           const size_t spitch, const size_t cols, 
                           const size_t rows,cudaStream_t &cq) {
  #ifdef UCL_DEBUG
  if (mat1::MEM_TYPE==1 && mat2::MEM_TYPE==0)
    assert(dst.kind()==UCL_RW_OPTIMIZED || dst.kind()==UCL_WRITE_OPTIMIZED);
  if (mat2::MEM_TYPE==1 && mat1::MEM_TYPE==0)
    assert(src.kind()==UCL_RW_OPTIMIZED || src.kind()==UCL_WRITE_OPTIMIZED);
  #endif
  _ucl_memcpy<mat1::MEM_TYPE,mat2::MEM_TYPE>::mc(dst.begin(),dpitch,src.begin(),
                                                 spitch,cols,rows,
                                                 _CUDA_TRAN<mat1::MEM_TYPE,
                                                         mat2::MEM_TYPE>::a(),
                                                 cq);
}

} // namespace ucl_cudart 

#endif

