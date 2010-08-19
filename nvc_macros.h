#ifndef NVC_MACROS_H
#define NVC_MACROS_H

#if defined(__APPLE__)
#if _GLIBCXX_ATOMIC_BUILTINS == 1
#undef _GLIBCXX_ATOMIC_BUILTINS
#endif // _GLIBCXX_ATOMIC_BUILTINS
#endif // __APPLE__

#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>

#ifndef UCL_NO_API_CHECK

#  define CUDA_SAFE_CALL( call) do {                                         \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        assert(0==1);                                                  \
    } } while (0)

#else  // not DEBUG

    // void macros for performance reasons
#  define CUDA_SAFE_CALL( call) call

#endif

#endif
