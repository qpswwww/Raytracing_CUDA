#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "cuda_error_handle.cuh"
#include "random.cuh"
#include "vec3.cuh"
#include "ray.cuh"
#include "world.cuh"
#include "hittable.cuh"
#include "camera.cuh"

__device__ vec3 color(const ray& r, hittable **world, curandState *pixel_random_seed) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, pixel_random_seed)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f*(unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns,camera **cam,hittable **world,curandState *random_seed) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = i * max_y + j;
	vec3 col(0, 0, 0);
	curandState pixel_random_seed = random_seed[pixel_index];
	//Montocaro random sampling
	for (int s = 0; s < ns; s++) {
		float u = float(i + curand_uniform(&pixel_random_seed))/float(max_x);
		float v = float(j + curand_uniform(&pixel_random_seed)) / float(max_y);
		ray r = (*cam)->get_ray(u, v); // , pixel_random_seed);
		col += color(r, world, &pixel_random_seed);
	}
	random_seed[pixel_index] = pixel_random_seed;
	col /= float(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
}

void GetImage(vec3* fb,int nx,int ny, int ns,char* save_dir,dim3 blocks, dim3 threads, camera **cam, hittable **world,curandState *random_seed) {
	FILE *stream1;
	freopen_s(&stream1, save_dir, "w", stdout);
	if (stream1 == NULL) {
		exit(9);
	}
	render<<<blocks, threads>>>(fb, nx, ny, ns, cam, world, random_seed);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			int pixel_idx = i*ny+j;
			float r = fb[pixel_idx][0];
			float g = fb[pixel_idx][1];
			float b = fb[pixel_idx][2];
			int ir = int(255.99*r);
			int ig = int(255.99*g);
			int ib = int(255.99*b);
			std::cout << ir << " " << ig << " " << ib << std::endl;
		}
	}
	/*size_t FB_size = nx*ny * sizeof(vec3); //the size of a frame buffer
	float *fb; //frame buffer
	checkCudaErrors(cudaMallocManaged((void**)&fb, FB_size));
	int tx = 8, ty = 8;
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render << <blocks, threads >> > (fb, nx, ny,50);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			int pixel_idx = j * 3 * nx + i * 3;
			float r = fb[pixel_idx + 0];
			float g = fb[pixel_idx + 1];
			float b = fb[pixel_idx + 2];
			int ir = int(255.99*r);
			int ig = int(255.99*g);
			int ib = int(255.99*b);
			std::cout << ir << " " << ig << " " << ib << std::endl;
		}
	}*/
	//checkCudaErrors(cudaFree(fb));
	fclose(stream1);
}

int main() {
	int dx, dy, tx, ty, ns;
	std::cin >> dx >> dy >> tx >> ty>>ns;
	std::cout << "dx=" << dx << " / dy=" << dy << " / tx=" << tx << " / ty=" << ty << " / ns=" << ns << std::endl;
	//int dx = 800, dy = 400, tx = 8, ty = 8,ns=100;
	dim3 blocks(dx / tx + 1, dy / ty + 1), threads(tx, ty);

	//Allocate the frame buffer
	vec3 *fb; //frame buffer
	checkCudaErrors(cudaMallocManaged((void**)&fb, dx*dy*sizeof(vec3)));

	//Allocate the memory of random seeds
	curandState *random_seed;
	//checkCudaErrors(cudaMalloc((void**)&random_seed,sizeof(random_seed)));
	checkCudaErrors(cudaMalloc((void**)&random_seed, dx*dy * sizeof(curandState)));
	init_pixel_random_seed << <blocks, threads >> >(dx, dy, random_seed);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	//init_random(dx,dy,random_seed,blocks,threads);
	
	camera** cam;
	checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera*)));

	hittable** obj_list;
	checkCudaErrors(cudaMalloc((void**)&obj_list, 2*sizeof(hittable*)));
	hittable** world;
	checkCudaErrors(cudaMalloc((void**)&world, sizeof(hittable*)));

	create_world<<<1,1>>>(obj_list,world,cam);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	GetImage(fb,dx,dy,100,"E://code//Raytracing_CUDA//output.ppm",blocks,threads,cam,world,random_seed);
	
	free_world << <1, 1 >> > (obj_list, world, cam);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(cam));
	checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(obj_list));
	checkCudaErrors(cudaFree(random_seed));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
	
	return 0;
}