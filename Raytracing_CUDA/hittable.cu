#include "hittable.cuh"
#include "ray.cuh"
#include "random.cuh"
#include "vec3.cuh"
#include "aabb.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "triangle.cuh"
#include "bvh.cuh"

__device__ bool hittable::hittable::hit(
	const ray& r, float t_min, float t_max, hit_record& rec) {
	switch (type) {
	case(type_hittable_list):
		return ((hittable_list*)this)->hit(r, t_min, t_max, rec);
		break;
	case(type_sphere):
		return ((sphere*)this)->hit(r, t_min, t_max, rec);
		break;
	case(type_triangle):
		return ((triangle*)this)->hit(r, t_min, t_max, rec);
		break;
	case(type_bvh_node):
		return ((bvh_node*)this)->hit(r, t_min, t_max, rec);
		break;
	}
	return false;
}

__host__ __device__ bool hittable::bounding_box(float t0, float t1, aabb& box) {
	switch (type) {
	case(type_hittable_list):
		return ((hittable_list*)this)->bounding_box(t0, t1, box);
		break;
	case(type_sphere):
		return ((sphere*)this)->bounding_box(t0, t1, box);
		break;
	case(type_triangle):
		return ((triangle*)this)->bounding_box(t0, t1, box);
		break;
	case(type_bvh_node):
		return ((bvh_node*)this)->bounding_box(t0, t1, box);
		break;
	}
	return false;
}