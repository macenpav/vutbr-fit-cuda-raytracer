#ifndef MATHEMATICS_H
#define MATHEMATICS_H

#include <cmath>

// device functions

namespace CUDA {

	__device__ inline float dot(float3 const& v1, float3 const& v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	
	__device__ inline float3 cross(float3 const& v1, float3 const& v2)
	{	
		return make_float3(v1.y * v2.z - v1.z * v2.y,
			v1.z * v2.x - v1.x * v2.z,
			v1.x * v2.y - v1.y * v2.x);	
	}

	__device__ inline float length(float3 const& vector) {
		return sqrt((vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z));
	}

	__device__ inline float cumax(float const& a, float const& b) { return a > b ? a : b; }

	__device__ inline float cumin(float const& a, float const& b) { return a > b ? b : a; }
	
}


// host functions

inline float dot(float3 const& v1, float3 const& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

	
inline float3 cross(float3 const& v1, float3 const& v2)
{	
	return make_float3(v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x);	
}

__host__ __device__
inline float3& normalize(float3& vector)
{
	float len = sqrt((vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z));

	vector.x /= len;
	vector.y /= len;
	vector.z /= len;	

	return vector;
}
	



inline float3 operator- (float3 const& l, float3 const& r)
{ 
	return make_float3(l.x - r.x, l.y - r.y, l.z - r.z);	
}

inline float3 operator+ (float3 const& l, float3 const& r)
{ 
	return make_float3(l.x + r.x, l.y + r.y, l.z + r.z);	
}

inline float3 operator* (float const& l, float3 const& r)
{ 
	return make_float3(l * r.x, l * r.y, l * r.z);	
}

inline float3& operator*= (float3& l, float const& r)
{ 
	l.x *= r; l.y *= r; l.z *= r; return l;
}

namespace CUDA {
	__device__ inline float3 float3_sub (float3 const& l, float3 const& r)
	{ 
		return make_float3(l.x - r.x, l.y - r.y, l.z - r.z);	
	}

	__device__ inline float3 float3_add (float3 const& l, float3 const& r)
	{ 
		return make_float3(l.x + r.x, l.y + r.y, l.z + r.z);	
	}

	__device__ inline float3 float3_add (float3 const& n1, float3 const& n2, float3 const& n3)
	{ 
		return make_float3(n1.x + n2.x + n3.x, n1.y + n2.y + n3.y, n1.z + n2.z + n3.z);	
	}

	__device__ inline float3 float3_mult (float const& l, float3 const& r)
	{ 
		return make_float3(l * r.x, l * r.y, l * r.z);	
	}

	__device__ inline float3& float3_multassign(float3& l, float const& r)
	{ 
		l.x *= r; l.y *= r; l.z *= r; return l;
	}
	__device__ inline float norm(float a) {
		if (a > 1.f)
		{
			return 1.0;
		}else
		{
			return a;
		}
	}

}

#endif
