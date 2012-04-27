/***************************************************************************
                                 ucl_test.cu
                             -------------------
                               W. Michael Brown

  Test driver for coprocessor library

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Wed Jan 28 2009
    copyright            : (C) 2009 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#include "nvc_device.h"
#include "nvd_device.h"
#include "ocl_device.h"
#include "nvc_mat.h"
#include "nvd_mat.h"
#include "ocl_mat.h"
#include "nvc_timer.h"
#include "nvd_timer.h"
#include "ocl_timer.h"
#if CUDART_VERSION >= 4000
#include "nvc_kernel.h"
#endif
#include "nvd_kernel.h"
#include "ocl_kernel.h"
#include "nvc_texture.h"
#include "nvd_texture.h"
#include "ocl_texture.h"
#include <cassert>
#include <sstream>

using namespace std;

const char * ocl_string_test = \
  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n \
   __kernel void vec_add(__global Scalar *a, __global Scalar *b, \
                         __global Scalar *ans) { \
     Ordinal i=get_global_id(0); \
     ans[i]=a[i]+b[i]; \
   }";
   
const char * nvd_string_bin = \
  "architecture {sm_10} \
   abiversion   {1} \
   modname      {cubin} \
   code { \
           name = vec_add \
           lmem = 0 \
           smem = 28 \
           reg  = 4 \
           bar  = 0 \
           bincode { \
                   0xa0004c05 0x04200780 0xa0004209 0x04200780 \
                   0x40020205 0x00018780 0xa0000001 0x04000780 \
                   0x20000201 0x04000780 0x30020009 0xc4100780 \
                   0x2102e800 0x2102ea0c 0xd00e0005 0x80c00780 \
                   0xd00e0601 0x80c00780 0xb0000204 0x2102ec00 \
                   0xd00e0005 0xa0c00781 \
           } \
   }";

const char * nvd_string_10_64 = \
  "     .version 1.4 \n\
        .target sm_10, map_f64_to_f32 \n\
        .entry vec_add ( \n\
                .param .u64 __cudaparm_vec_add_a, \n\
                .param .u64 __cudaparm_vec_add_b, \n\
                .param .u64 __cudaparm_vec_add_ans) \n\
        { \n\
        .reg .u32 %r<7>; \n\
        .reg .u64 %rd<10>; \n\
        .reg .f32 %f<5>; \n\
        .loc    27      37      0 \n\
   $LBB1_vec_add: \n\
        .loc    27      38      0 \n\
        cvt.s32.u16     %r1, %ctaid.x; \n\
        cvt.s32.u16     %r2, %ntid.x; \n\
        mul24.lo.s32    %r3, %r1, %r2; \n\
        cvt.u32.u16     %r4, %tid.x; \n\
        add.u32         %r5, %r3, %r4; \n\
        cvt.u64.s32     %rd1, %r5; \n\
        mul.lo.u64      %rd2, %rd1, 4; \n\
        ld.param.u64    %rd3, [__cudaparm_vec_add_a]; \n\
        add.u64         %rd4, %rd3, %rd2; \n\
        ld.global.f32   %f1, [%rd4+0]; \n\
        ld.param.u64    %rd5, [__cudaparm_vec_add_b]; \n\
        add.u64         %rd6, %rd5, %rd2; \n\
        ld.global.f32   %f2, [%rd6+0]; \n\
        add.f32         %f3, %f1, %f2; \n\
        ld.param.u64    %rd7, [__cudaparm_vec_add_ans]; \n\
        add.u64         %rd8, %rd7, %rd2; \n\
        st.global.f32   [%rd8+0], %f3; \n\
        exit; \n\
   $LDWend_vec_add: \n\
        }\n";

template <class mat_type>
void fill_test(mat_type &mat, const size_t n) {
  for (size_t i=0; i<n; i++)
    mat[i]=static_cast<typename mat_type::data_type>(i);
  for (size_t i=0; i<n; i++)
    assert(mat[i]==static_cast<typename mat_type::data_type>(i));
}

template <class mat_type>
void fill_test(mat_type &mat, const size_t rows, const size_t cols) {
  for (size_t i=0; i<cols; i++)
    for (size_t j=0; j<rows; j++)
      mat(j,i)=static_cast<typename mat_type::data_type>(i*rows+j);
  for (size_t i=0; i<cols; i++)
    for (size_t j=0; j<rows; j++)
      assert(mat(j,i)==static_cast<typename mat_type::data_type>(i*rows+j));
}

template <class numtyp>
void cudadr_test(const bool async, const string cubin_dir) {
  using namespace ucl_cudadr;
  #include "ucl_test_source.h"  
  string kernel_name=cubin_dir+string("/ucl_test_kernel.ptx");  
  const char *kernel_string=nvd_string_10_64;
  if (sizeof(numtyp)==8)
    kernel_name=cubin_dir+string("/ucl_test_kernel_d.ptx");
  #include "ucl_test_vecadd.h"  
  cerr << "DONE.\n";
}

template <class numtyp>
void cudart_test(const bool async) {
  using namespace ucl_cudart;
  #if CUDART_VERSION >= 4000
  #include "ucl_test_source.h"
  string kernel_name="vec_add";
  const char *kernel_string="vec_add";
  if (sizeof(numtyp)!=8) {
    #include "ucl_test_vecadd.h"
  }
  #else
  cerr << "NOT TESTING CUDA RUNTIME KERNEL VEC ADD.\n";
  #endif
  cerr << "DONE.\n";
}

template <class numtyp>
void opencl_test(const bool async) {
  using namespace ucl_opencl;
  #include "ucl_test_source.h"
  string kernel_name="ucl_test_kernel.cu";  
  const char *kernel_string=ocl_string_test;
  #include "ucl_test_vecadd.h"  
  cerr << "DONE.\n";
}

int main(int argc, char** argv) {
  assert(argc==2);
  string cubin_dir=argv[1];

  #ifdef UCL_DEBUG
  cerr << "RUNNING ALL TESTS IN DEBUG MODE...\n\n";
  #endif

  cerr << "----------------------------------------------------------------\n";
  cerr << "|           Singe Precision CUDADR Blocking Tests              |\n";
  cerr << "----------------------------------------------------------------\n";
  cudadr_test<float>(false,cubin_dir);
  cerr << "\n\n";
  
  cerr << "----------------------------------------------------------------\n";
  cerr << "|         Singe Precision CUDADR Asynchronous Tests            |\n";
  cerr << "----------------------------------------------------------------\n";
  cudadr_test<float>(true,cubin_dir);
  cerr << "\n\n";

  cerr << "----------------------------------------------------------------\n";
  cerr << "|           Double Precision CUDADR Blocking Tests             |\n";
  cerr << "----------------------------------------------------------------\n";
  cudadr_test<double>(false,cubin_dir);
  cerr << "\n\n";
  
  cerr << "----------------------------------------------------------------\n";
  cerr << "|         Double Precision CUDADR Asynchronous Tests           |\n";
  cerr << "----------------------------------------------------------------\n";
  cudadr_test<double>(true,cubin_dir);
  cerr << "\n\n";

  cerr << "----------------------------------------------------------------\n";
  cerr << "|           Singe Precision CUDART Blocking Tests              |\n";
  cerr << "----------------------------------------------------------------\n";
  cudart_test<float>(false);
  cerr << "\n\n";
  
  cerr << "----------------------------------------------------------------\n";
  cerr << "|         Singe Precision CUDART Asynchronous Tests            |\n";
  cerr << "----------------------------------------------------------------\n";
  cudart_test<float>(true);
  cerr << "\n\n";

  cerr << "----------------------------------------------------------------\n";
  cerr << "|           Double Precision CUDART Blocking Tests             |\n";
  cerr << "----------------------------------------------------------------\n";
  cudart_test<double>(false);
  cerr << "\n\n";
  
  cerr << "----------------------------------------------------------------\n";
  cerr << "|         Double Precision CUDART Asynchronous Tests           |\n";
  cerr << "----------------------------------------------------------------\n";
  cudart_test<double>(true);
  cerr << "\n\n";

  cerr << "----------------------------------------------------------------\n";
  cerr << "|            Single Precision OpenCL Blocking Tests            |\n";
  cerr << "----------------------------------------------------------------\n";
  opencl_test<float>(false);
  cerr << "\n\n";

  cerr << "----------------------------------------------------------------\n";
  cerr << "|          Single Precision OpenCL Asynchronous Tests          |\n";
  cerr << "----------------------------------------------------------------\n";
  opencl_test<float>(true);
  cerr << "\n\n";

  cerr << "----------------------------------------------------------------\n";
  cerr << "|            Double Precision OpenCL Blocking Tests            |\n";
  cerr << "----------------------------------------------------------------\n";
  opencl_test<double>(false);
  cerr << "\n\n";

  cerr << "----------------------------------------------------------------\n";
  cerr << "|          Double Precision OpenCL Asynchronous Tests          |\n";
  cerr << "----------------------------------------------------------------\n";
  opencl_test<double>(true);
  cerr << "\n\n";

  cerr << "ALL TESTS PASSED.\n\n\n";
  
  return 0;
}
