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

//#include "ocl_device.h"
//#include "ocl_mat.h"
//#include "ocl_timer.h"
//#include "ocl_kernel.h"
//#include "ocl_texture.h"
#include "geryon.h"
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
void opencl_test(const bool async, const string cubin_dir) {
  using namespace ucl_opencl;
  #include "ucl_test_source.h"
  string kernel_name=cubin_dir+string("/ucl_test_kernel.cu");  
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
  cerr << "|            Single Precision OpenCL Blocking Tests            |\n";
  cerr << "----------------------------------------------------------------\n";
  opencl_test<float>(false,cubin_dir);
  cerr << "\n\n";

  cerr << "----------------------------------------------------------------\n";
  cerr << "|          Single Precision OpenCL Asynchronous Tests          |\n";
  cerr << "----------------------------------------------------------------\n";
  opencl_test<float>(true,cubin_dir);
  cerr << "\n\n";

  cerr << "----------------------------------------------------------------\n";
  cerr << "|            Double Precision OpenCL Blocking Tests            |\n";
  cerr << "----------------------------------------------------------------\n";
  opencl_test<double>(false,cubin_dir);
  cerr << "\n\n";

  cerr << "----------------------------------------------------------------\n";
  cerr << "|          Double Precision OpenCL Asynchronous Tests          |\n";
  cerr << "----------------------------------------------------------------\n";
  opencl_test<double>(true,cubin_dir);
  cerr << "\n\n";

  cerr << "ALL TESTS PASSED.\n\n\n";
  
  return 0;
}
