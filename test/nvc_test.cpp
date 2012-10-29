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

//#include "nvc_device.h"
//#include "nvc_mat.h"
//#include "nvc_timer.h"
//#if CUDART_VERSION >= 4000
//#include "nvc_kernel.h"
//#endif
//#include "nvc_texture.h"
#include "geryon.h"
#include <cassert>
#include <sstream>

using namespace std;

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


int main(int argc, char** argv) {
  assert(argc==2);
  string cubin_dir=argv[1];

  #ifdef UCL_DEBUG
  cerr << "RUNNING ALL TESTS IN DEBUG MODE...\n\n";
  #endif

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

  cerr << "ALL TESTS PASSED.\n\n\n";
  
  return 0;
}
