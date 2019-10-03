#ifndef RANDOM_CUH
#define RANDOM_CUH

#include <curand_kernel.h>
#include "cuda_error_handle.cuh"
#include "vec3.cuh"

//Initialize the random seed for each pixel
__global__ void init_pixel_random_seed(int max_x, int max_y, curandState *random_seed) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_idx = i * max_y + j;
	curand_init(1999, pixel_idx, 0, &random_seed[pixel_idx]);
}
/*
void init_random(int max_x, int max_y,curandState **random_seed,dim3 &blocks,dim3 &threads) {
	checkCudaErrors(cudaMalloc((void**)random_seed,max_x*max_y*sizeof(random_seed)));
	init_pixel_random_seed<<<blocks,threads>>>(max_x,max_y,random_seed);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
*/
/*
inline double random_double() {
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);
	static std::mt19937 generator;
	static std::function<double()> rand_generator =
		std::bind(distribution, generator);
	return rand_generator();
}*/

//Generate a vector in an unit sphere according to the pixel's random seed
__device__ inline vec3 random_in_unit_sphere(curandState *random_seed) {
	vec3 p;
	do {
		p = 2.0*vec3(curand_uniform(random_seed), curand_uniform(random_seed), curand_uniform(random_seed)) - vec3(1, 1, 1);
	} while (p.squared_length() >= 1.0);
	return p;
}

#endif // !RANDOM_CUH
