/***************************************************************************
                                ucl_doxygen.h
                             -------------------
                               W. Michael Brown

  Doxygen documentation for Geryon

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Mon Mar 13 2010
    copyright            : (C) 2010 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2010) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */
/**
 * \file geryon.h 
 * \brief Main geryon header 
 * \mainpage Geryon
 * \section About
 * Geryon is a simple C++ library intended to simplify device and memory
 * management, I/O, and timing for the CUDA and OpenCL APIs. Additionally, 
 * it provides a single interface to the CUDA-Driver, CUDA-Runtime, and 
 * OpenCL APIs to allow a single code to compile using any of the 3 APIs.
 *
 * The Geryon source code, examples, and slides describing Geryon and
 * CUDA and OpenCL APIs can be found here:
 *
 * http://scicomp.github.com/geryon
 *
 * Geryon is divided into 3 namespaces:
 *   - ucl_cudart - Namespace for CUDA-Runtime
 *   - ucl_cudadr - Namespace for CUDA-OpenCL
 *   - ucl_opencl - Namespace for OpenCL
 *
 * With only a few exceptions, the classes and (typedefed) prototypes 
 * will be the same for all 3 namespaces. The doxygen documentation
 * is probably best navigated by starting with one of the 3 namespaces.
 * <p>
 * Copyright (2010) Sandia Corporation.  Under the terms of Contract 
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains 
 * certain rights in this software.  This software is distributed under 
 * the Simplified BSD License.</p>
 */
  
#ifndef GERYON_H
#define GERYON_H

#include <string>

#if !defined(UCL_OPENCL) && !defined(UCL_CUDADR) && !defined(UCL_CUDART)
#error "Must define a target platform"
#endif

// OpenCL headers
#ifdef UCL_OPENCL
#include "ocl_device.h"
#include "ocl_mat.h"
#include "ocl_kernel.h"
#include "ocl_macros.h"
#include "ocl_memory.h"
#include "ocl_texture.h"
#include "ocl_timer.h"
using namespace ucl_opencl;
#endif

// CUDA driver headers
#ifdef UCL_CUDADR
#include "nvd_device.h"
#include "nvd_mat.h"
#include "nvd_kernel.h"
#include "nvd_macros.h"
#include "nvd_memory.h"
#include "nvd_texture.h"
#include "nvd_timer.h"
using namespace ucl_cudadr;
#endif

// CUDA runtime headers
#ifdef UCL_CUDART
#include "nvc_device.h"
#include "nvc_mat.h"
#include "nvc_kernel.h"
#include "nvc_macros.h"
#include "nvc_memory.h"
#include "nvc_texture.h"
#include "nvc_timer.h"
#include "nvc_traits.h"
using namespace ucl_cudart;
#endif

// Standard ucl headers
#include "ucl_basemat.h"
#include "ucl_copy.h"
#include "ucl_d_mat.h"
#include "ucl_d_vec.h" 
#include "ucl_h_mat.h" 
#include "ucl_h_vec.h" 
#include "ucl_image.h"
#include "ucl_matrix.h"
#include "ucl_nv_kernel.h"
#include "ucl_print.h"
#include "ucl_types.h" 
#include "ucl_vector.h"
#include "ucl_version.h"

/** 
 * \brief Converts to human readable error. 
 * \param result Code that has been returned by geryon call.
 * \return The string corresponding to the result code.
 */
inline std::string ucl_check(int result)
{
   switch(result) {
      case UCL_ERROR: 
         return std::string("UCL_ERROR"); break; 
      case UCL_SUCCESS: 
         return std::string("UCL_SUCCESS"); break; 
      case UCL_COMPILE_ERROR: 
         return std::string("UCL_COMPILE_ERROR"); break; 
      case UCL_FILE_NOT_FOUND: 
         return std::string("UCL_FILE_NOT_FOUND"); break; 
      case UCL_FUNCTION_NOT_FOUND: 
         return std::string("UCL_FUNCTION_NOT_FOUND"); break;
      case UCL_MEMORY_ERROR: 
         return std::string("UCL_MEMORY_ERROR"); break; 
   }
   return std::string("Unknown");
}

#endif
