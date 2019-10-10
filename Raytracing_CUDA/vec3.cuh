#ifndef VEC3_CUH
#define VEC3_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>

__host__ __device__ float ffmin(float a, float b);
__host__ __device__ float ffmax(float a, float b);

class vec3 {
public:
	__host__ __device__ vec3();
	__host__ __device__ vec3(float e0, float e1, float e2);
	__host__ __device__ float x() const;
	__host__ __device__ float y() const;
	__host__ __device__ float z() const;
	__host__ __device__ float r() const;
	__host__ __device__ float g() const;
	__host__ __device__ float b() const;

	__host__ __device__ const vec3& operator+() const;
	__host__ __device__ vec3 operator-() const;
	__host__ __device__ float operator[](int i) const;
	__host__ __device__ float& operator[](int i);

	__host__ __device__ vec3& operator+=(const vec3 &v2);
	__host__ __device__ vec3& operator-=(const vec3 &v2);
	__host__ __device__ vec3& operator*=(const vec3 &v2);
	__host__ __device__ vec3& operator/=(const vec3 &v2);
	__host__ __device__ vec3& operator*=(const float t);
	__host__ __device__ vec3& operator/=(const float t);

	__host__ __device__ float length() const;
	__host__ __device__ float squared_length() const;
	__host__ __device__ void make_unit_vector();

	float e[3];
};

std::istream& operator>>(std::istream &is, vec3 &t);

std::ostream& operator<<(std::ostream &os, const vec3 &t);

__host__ __device__ vec3 operator+(const vec3 &v1, const vec3 &v2);

__host__ __device__ vec3 operator-(const vec3 &v1, const vec3 &v2);

__host__ __device__ vec3 operator*(const vec3 &v1, const vec3 &v2);

__host__ __device__ vec3 operator*(float t, const vec3 &v);

__host__ __device__ vec3 operator*(const vec3 &v, float t);

__host__ __device__ vec3 operator/(const vec3 &v1, const vec3 &v2);

__host__ __device__ vec3 operator/(vec3 v, float t);

__host__ __device__ float dot(const vec3 &v1, const vec3 &v2);

__host__ __device__ vec3 cross(const vec3 &v1, const vec3 &v2);

__host__ __device__ vec3 unit_vector(vec3 v);

__host__ __device__ vec3 reflect(const vec3& v, const vec3& n);

__host__ __device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted);

__host__ __device__ float schlick(float cosine, float ref_idx);

#endif // !VEC3H