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

/*! \file */

/** \mainpage Geryon
  * \section About
  * Geryon is a simple C++ library intended to simplify device and memory
  * management, I/O, and timing for the CUDA and OpenCL APIs. Additionally, 
  * it provides a single interface to the CUDA-Driver, CUDA-Runtime, and 
  * OpenCL APIs to allow a single code to compile using any of the 3 APIs.
  *
  * The Geryon source code, examples, and slides describing Geryon and
  * CUDA and OpenCL APIs can be found here:
  *
  * http://users.nccs.gov/~wb8/geryon/index.htm
  *
  * Geryon is divided into 3 namespaces:
  *   - ucl_cudart - Namespace for CUDA-Runtime
  *   - ucl_cudadr - Namespace for CUDA-OpenCL
  *   - ucl_opencl - Namespace for OpenCL
  *
  * With only a few exceptions, the classes and (typedefed) prototypes 
  * will be the same for all 3 namespaces. The doxygen documentation
  * is probably best navigated by starting with one of the 3 namespaces.
  *
  **/

