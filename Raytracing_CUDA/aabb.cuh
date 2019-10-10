#ifndef AABB_CUH
#define AABB_CUH

#include "vec3.cuh"
#include "ray.cuh"

class aabb {
public:
	__host__ __device__ aabb();
	__host__ __device__ aabb(const vec3& a, const vec3& b);

	__host__ __device__ vec3 min() const;
	__host__ __device__ vec3 max() const;

	__device__ bool hit(const ray& r, float tmin, float tmax);

	vec3 _min;
	vec3 _max;
};

//compute the bounding box of two boxes on the fly
__host__ __device__ aabb surrounding_box(aabb box0, aabb box1);

#endif