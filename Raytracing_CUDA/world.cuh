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

__global__ void create_world(hittable **obj_list, hittable **world, camera **cam) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		obj_list[0] = new sphere(vec3(0, 0, -1), 0.5,
			new lambertian(vec3(0.1, 0.2, 0.5)));

		obj_list[1] = new sphere(vec3(0, -100.5, -1), 100,
			new lambertian(vec3(0.8, 0.8, 0.0)));

		obj_list[2] = new sphere(vec3(1, 0, -1), 0.5,
			new metal(vec3(0.8, 0.6, 0.2), 0.0));

		obj_list[3] = new sphere(vec3(-1, 0, -1), 0.5,
			new dielectric(1.5));

		obj_list[4] = new sphere(vec3(-1, 0, -1), -0.45,
			new dielectric(1.5));
		
		*world = new hittable_list(obj_list,5);
		*cam = new camera();
	}
}

__global__ void free_world(hittable **obj_list, hittable **world, camera **cam) {
	for (int i = 0; i < ((hittable_list*)(*world))->list_size; i++) {
		delete ((sphere*)((*obj_list) + i))->mat_ptr;
		delete ((*obj_list) + i);
	}

	delete (*world);
	delete (*cam);
}

#endif