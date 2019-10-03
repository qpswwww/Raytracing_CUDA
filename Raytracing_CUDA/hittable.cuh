#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "ray.cuh"
#include "random.cuh"

class hit_record;

class material {
public:
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation,
		ray& scattered, curandState *pixel_random_seed) const = 0;
};

class hit_record {
public:
	float t;
	vec3 p;
	vec3 normal;
	material *mat_ptr;
};

class hittable {
public:
	__device__ virtual bool hit(
		const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif