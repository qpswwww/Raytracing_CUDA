#include "material.cuh"
#include "hittable.cuh"
#include "random.cuh"

__device__ bool material::scatter(const ray& r_in, const hit_record& rec, vec3& attenuation,
	ray& scattered, curandState *pixel_random_seed) {
	switch (type) {
	case(type_lambertian):
		return ((lambertian*)this)->scatter(r_in, rec, attenuation, scattered, pixel_random_seed);
		break;
	case(type_metal):
		return ((metal*)this)->scatter(r_in, rec, attenuation, scattered, pixel_random_seed);
		break;
	case(type_dielectric):
		return ((dielectric*)this)->scatter(r_in, rec, attenuation, scattered, pixel_random_seed);
		break;
	}
	return false;
}

__host__ __device__ lambertian::lambertian(const vec3& a) : albedo(a) {
	type = type_lambertian;
}
	__device__ bool lambertian::scatter(const ray& r_in, const hit_record& rec,
		vec3& attenuation, ray& scattered, curandState *pixel_random_seed) {
		vec3 target = rec.p + rec.normal + random_in_unit_sphere(pixel_random_seed);
		scattered = ray(rec.p, target - rec.p);
		attenuation = albedo;
		return true;
	}

	__host__ __device__ metal::metal(const vec3& a, float f) : albedo(a) {
		type = type_metal;
		if (f < 1) fuzz = f; else fuzz = 1;
	}

	__device__ bool metal::scatter(const ray& r_in, const hit_record& rec,
		vec3& attenuation, ray& scattered, curandState *pixel_random_seed) {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(pixel_random_seed));
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}

	__host__ __device__ dielectric::dielectric(float ri) : ref_idx(ri) {
		type = type_dielectric;
	}

	__device__ bool dielectric::scatter(const ray& r_in, const hit_record& rec,
		vec3& attenuation, ray& scattered, curandState *pixel_random_seed) {
		vec3 outward_normal;
		vec3 reflected = reflect(r_in.direction(), rec.normal);
		float ni_over_nt;
		attenuation = vec3(1.0, 1.0, 1.0);
		vec3 refracted;

		float reflect_prob;
		float cosine;

		if (dot(r_in.direction(), rec.normal) > 0) {
			outward_normal = -rec.normal;
			ni_over_nt = ref_idx;
			cosine = ref_idx * dot(r_in.direction(), rec.normal)
				/ r_in.direction().length();
		}
		else {
			outward_normal = rec.normal;
			ni_over_nt = 1.0 / ref_idx;
			cosine = -dot(r_in.direction(), rec.normal)
				/ r_in.direction().length();
		}

		if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
			reflect_prob = schlick(cosine, ref_idx);
		}
		else {
			reflect_prob = 1.0;
		}

		if (curand_uniform(pixel_random_seed) < reflect_prob) {
			scattered = ray(rec.p, reflected);
		}
		else {
			scattered = ray(rec.p, refracted);
		}

		return true;
	}