#ifndef OCL_MACROS_H
#define OCL_MACROS_H

#include <stdio.h>
#include <cassert>
#include "CL/cl.h"

#ifndef UCL_NO_API_CHECK

#  define CL_SAFE_CALL( call) do {                                         \
    cl_int err = call;                                                     \
    if( err != CL_SUCCESS) {                                               \
        fprintf(stderr, "OpenCL error in file '%s' in line %i : %d.\n",    \
                __FILE__, __LINE__, err );                                 \
        assert(0==1);                                                           \
    } } while (0)

#  define CL_CHECK_ERR( val) do {                                        \
    if( val != CL_SUCCESS) {                                               \
        fprintf(stderr, "OpenCL error in file '%s' in line %i : %d.\n",    \
                __FILE__, __LINE__, val );                                 \
        assert(0==1);                                                           \
    } } while (0)

#else  // not DEBUG

// void macros for performance reasons
#  define CL_SAFE_CALL( call) call
#  define CL_CHECK_ERR( val)

#endif

#endif
