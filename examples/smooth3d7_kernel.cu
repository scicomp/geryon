#ifdef NV_KERNEL

#define __global  
#define __kernel extern "C" __global__
#define SYNC __syncthreads
#define SHARED __shared__

#else

#define blockIdx.x get_group_id(0)
#define blockIdx.y get_group_id(1)
#define blockIdx.z get_group_id(2)

#define blockDim.x get_num_groups(0)
#define blockDim.y get_num_groups(1)
#define blockDim.z get_num_groups(2)

#define threadIdx.x get_local_id(0)
#define threadIdx.y get_local_id(1)
#define threadIdx.z get_local_id(2)

#define SYNC barrier
#define SHARED __local

#endif

#define TYPE float
#define BLOCK_SIZE 512

__kernel void smooth3d7(__global TYPE *u, __global TYPE* v, __global  TYPE* A, __global TYPE* f)
{   
//   SHARED TYPE sres[BLOCK_SIZE];
//   
//   // Global 3D index of the thread in the physical domain
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   int j = blockIdx.y * blockDim.y + threadIdx.y;
//   int k = blockIdx.z * blockDim.z + threadIdx.z;

//   // Global 1D index of the thread in the physical domain w/o ghost
//   int gid = k * gridDim.x * blockDim.x * gridDim.y * blockDim.y 
//           + j * gridDim.x * blockDim.x + i; 

//   // Local 1D index of the thread in the thread block 
//   int lid = threadIdx.z * blockDim.y * blockDim.x  
//           + threadIdx.y * blockDim.x + threadIdx.x;
//   
//   // Block size of a block in device/physical domain
//   int n = blockDim.x * blockDim.y * blockDim.z;
//   int nx = gridDim.x * blockDim.x + 2;
//   int ny = gridDim.y * blockDim.y + 2;
//   int xy = nx * ny;

//   // Set residual with right hand side
//   double r = f[gid];

//   // Shift thread index by halos
//   k += 1; j += 1; i += 1;

//   // Apply the neighbour values in respect to stencil
//   r -= -1 * u[(k+1)*xy + (j)*nx   + (i)]; 
//   r -= -1 * u[(k-1)*xy + (j)*nx   + (i)];
//   r -= -1 * u[(k)*xy   + (j+1)*nx + (i)];
//   r -= -1 * u[(k)*xy   + (j-1)*nx + (i)];
//   r -= -1 * u[(k)*xy   + (j)*nx   + (i+1)];  
//   r -= -1 * u[(k)*xy   + (j)*nx   + (i-1)]; 
//   r -=  6 * u[(k)*xy   + (j)*nx   + (i)]; 
//   gid = (k)*xy + (j)*nx + (i);

//   // Update shared memory
//   sres[lid] = r; r = 0;
//   SYNC();

//   // Multiply the block with the inverse
//   for(int g = 0; g < n; g++)
//      r += A[g*n+lid] * sres[g];

//   // Advance solution    
//   v[gid] = 0.8 * r + u[gid];
}
