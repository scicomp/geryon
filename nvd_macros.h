#ifndef NVD_MACROS_H
#define NVD_MACROS_H

#include <stdio.h>
#include <cassert>
#include <cuda.h>

#ifndef UCL_NO_API_CHECK

#  define CU_SAFE_CALL( call ) do {                                          \
    CUresult err = call;                                                     \
    if( CUDA_SUCCESS != err) {                                               \
        fprintf(stderr, "Cuda driver error %d in file '%s' in line %i.\n",   \
                err, __FILE__, __LINE__ );                                   \
        assert(0==1);                                                  \
    } } while (0)

#else  // not DEBUG

    // void macros for performance reasons
#  define CU_SAFE_CALL( call) call

#endif

#endif
