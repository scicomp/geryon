// For testing the example compile only
const char * kernel_string = 
"#ifdef NV_KERNEL\n"
"#define __global  \n"
"#define GLOBAL_ID_X threadIdx.x+blockIdx.x*blockDim.x\n"
"#define __kernel extern \"C\" __global__\n"
"#else\n"
"#define GLOBAL_ID_X get_global_id(0)\n"
"#endif\n"
"#define Scalar float\n"
"__kernel void vec_add(__global Scalar *a, __global Scalar *b, \n"
"                      __global Scalar *ans) {\n"
"  int i=GLOBAL_ID_X;\n"
"  ans[i]=a[i]+b[i];\n"
"}\n"
;
