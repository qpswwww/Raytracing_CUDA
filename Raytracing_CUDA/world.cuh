#ifndef WORLD_CUH
#define WORLD_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "cuda_error_handle.cuh"
#include "material.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include <stdio.h>

__global__ void create_world(hittable **obj_list, hittable **world, camera **cam, int dx,int dy,curandState *random_seed) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		/*obj_list[0] = new sphere(vec3(0, 0, -1), 0.5,
			new lambertian(vec3(0.1, 0.2, 0.5)));

		obj_list[1] = new sphere(vec3(0, -100.5, -1), 100,
			new lambertian(vec3(0.8, 0.8, 0.0)));

		obj_list[2] = new sphere(vec3(1, 0, -1), 0.5,
			new metal(vec3(0.8, 0.6, 0.2), 0.0));

		obj_list[3] = new sphere(vec3(-1, 0, -1), 0.5,
			new dielectric(1.5));

		obj_list[4] = new sphere(vec3(-1, 0, -1), -0.45,
			new dielectric(1.5));

		*world = new hittable_list(obj_list, 5);
		*/
		//int n = 500;
		//hittable **list = new hittable*[n + 1];

		/*curandState *pixel_random_seed = &(random_seed[0]);
		int i = 0;
		obj_list[i++] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
		for (int a = -10; a < 10; a++) {
			for (int b = -10; b < 10; b++) {
				float choose_mat = curand_uniform(pixel_random_seed);
				vec3 center(a + 0.9f*curand_uniform(pixel_random_seed), 0.2, b + 0.9f*curand_uniform(pixel_random_seed));
				if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
					if (choose_mat < 0.8) {  // diffuse
						obj_list[i++] = new sphere(center, 0.2,
							new lambertian(vec3(curand_uniform(pixel_random_seed)*curand_uniform(pixel_random_seed),
								curand_uniform(pixel_random_seed)*curand_uniform(pixel_random_seed),
								curand_uniform(pixel_random_seed)*curand_uniform(pixel_random_seed))
							)
						);
					}
					else if (choose_mat < 0.95) { // metal
						obj_list[i++] = new sphere(center, 0.2,
							new metal(vec3(0.5f*(1 + curand_uniform(pixel_random_seed)),
								0.5f*(1 + curand_uniform(pixel_random_seed)),
								0.5f*(1 + curand_uniform(pixel_random_seed))),
								0.5f*curand_uniform(pixel_random_seed)));
					}
					else {  // glass
						obj_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
					}
				}
			}
		}

		obj_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
		obj_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
		obj_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
		
		*world = new hittable_list(obj_list, i);
		*/
		vec3 lookfrom(100, 100, 100);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10;
		float aperture = 0.1;

		*cam = new camera(lookfrom, lookat, vec3(0, 1, 0), 70,
			float(dx) / float(dy), aperture, dist_to_focus);
	}
}

__global__ void free_world(hittable **obj_list, hittable **world, camera **cam) {
	printf("ok1\n");
	for (int i = 0; i < ((hittable_list*)(*world))->list_size; i++) {
		delete ((sphere*)obj_list[i])->mat_ptr;
		delete obj_list[i];
	}
	printf("ok2\n"); 
	delete (*world);
	printf("ok3\n"); 
	delete (*cam);
}

#endif