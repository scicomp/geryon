#ifndef GERYON_H
#define GERYON_H

#ifdef UCL_OPENCL
#include "ocl_device.h"
#include "ocl_mat.h"
using namespace ucl_opencl;
#endif

#ifdef UCL_CUDADR
#include "nvd_device.h"
#include "nvd_mat.h"
using namespace ucl_cudadr;
#endif

#ifdef UCL_CUDART
#include "nvc_device.h"
#include "nvc_mat.h"
using namespace ucl_cudart;
#endif

#endif
