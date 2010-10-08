/***************************************************************************
                                  nvc_mat.h
                             -------------------
                               W. Michael Brown

  CUDA Specific Vector/Matrix Containers, Memory Management, and I/O

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Thu Jun 25 2009
    copyright            : (C) 2009 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

/*! \file */
   
#ifndef NVC_MAT_H
#define NVC_MAT_H

#include "nvc_memory.h"

/// Namespace for CUDA Runtime routines
namespace ucl_cudart {

#define _UCL_MAT_ALLOW
#include "ucl_basemat.h"
#include "ucl_h_vec.h"
#include "ucl_h_mat.h"
#include "ucl_d_vec.h"
#include "ucl_d_mat.h"
#undef _UCL_MAT_ALLOW

#define UCL_COPY_ALLOW
#include "ucl_copy.h"
#undef UCL_COPY_ALLOW

#define UCL_PRINT_ALLOW
#include "ucl_print.h"
#undef UCL_PRINT_ALLOW

} // namespace ucl_cudart 

#endif
