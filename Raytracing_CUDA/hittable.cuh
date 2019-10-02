#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "ray.cuh"
#include "random.cuh"
#include "material.cuh"

class material;

struct hit_record {
	float t;
	vec3 p;
	vec3 normal;
	material *mat_ptr;
};

public class hittable {
public:
	virtual bool hit(
		const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif