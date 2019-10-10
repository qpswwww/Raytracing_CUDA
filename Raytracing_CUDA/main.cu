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
#include "bvh.cuh"
#include "obj_parser.cuh"

__device__ vec3 color(const ray& r, hittable **world, curandState *pixel_random_seed) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		aabb tmp;
		//bool xxx=((bvh_node*)(*world))->test(tmp);
		//((bvh_node*)(*world))->bounding_box(0.0f,1.0f,tmp);
		//((bvh_node*)(*world))->hit(cur_ray, 0.001f, FLT_MAX, rec);
		
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
		ray r = (*cam)->get_ray(u, v,&pixel_random_seed); // , pixel_random_seed);
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
	fclose(stream1);
}

__global__ void visit2(hittable** obj_list,int list_size) {
	hittable *p = (triangle*)obj_list[1];
	material* mp = p->mat_ptr;
	((dielectric*)mp)->ref_idx += 0.01;
}

int main() {
	int dx, dy, tx, ty, ns;
	std::cin >> dx >> dy >> tx >> ty>>ns;
	std::cout << "dx=" << dx << " / dy=" << dy << " / tx=" << tx << " / ty=" << ty << " / ns=" << ns << std::endl;
	//int dx = 800, dy = 400, tx = 8, ty = 8,ns=100;
	dim3 blocks(dx / tx + 1, dy / ty + 1), threads(tx, ty);

	hittable** obj_list_host = (hittable**)malloc((10000) * sizeof(hittable*));
	hittable** obj_list_tmp = (hittable**)malloc((10000) * sizeof(hittable*));

	int list_size = 0;
	read_obj_file((char*)"E:/code/cpp/teapot.obj", obj_list_host, list_size);
	
	//hittable** obj_list_device = copy_objs_to_gpu(obj_list_host,list_size);

	//visit2 << <1, 1 >> >(obj_list_device,list_size);
	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaDeviceSynchronize());
	

	hittable *world_host;
	world_host = new bvh_node(obj_list_host, 0, list_size, 0, 1, 0);

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

	hittable **world_device;
	checkCudaErrors(cudaMalloc((void**)&world_device, sizeof(hittable*)));

	hittable **tmp= new hittable*[1];
	*tmp=world_host->copy_to_gpu();
	checkCudaErrors(cudaMemcpy(world_device,tmp, sizeof(hittable*), cudaMemcpyHostToDevice));
	//create_world<<<1,1>>>(obj_list,world,cam,dx,dy,random_seed);
	int *count_device,*count_host=new int;
	checkCudaErrors(cudaMalloc((void**)&count_device, sizeof(int)));
	visit << <1, 1 >> >((bvh_node**)world_device,count_device);


	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(count_host, count_device, sizeof(int), cudaMemcpyDeviceToHost));
	std::cerr << "dbg:count=" << *count_host << std::endl;
	
	create_world << <1, 1 >> >(world_device, world_device, cam, dx, dy, random_seed);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	GetImage(fb,dx,dy,ns,"E://code//Raytracing_CUDA//output.ppm",blocks,threads,cam,world_device,random_seed);
	/*
	free_world << <1, 1 >> > (obj_list_device, world_device, cam);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(cam));
	checkCudaErrors(cudaFree(world_device));
	//checkCudaErrors(cudaFree(obj_list_device));
	checkCudaErrors(cudaFree(random_seed));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
	*/
	return 0;
}