#ifndef RAY_H
#define RAY_H

#include "mathematics.h"

struct Ray {
	__device__ Ray()	
	{}

	__device__ Ray(float3 const& o, float3 d)
		: origin(o), direction(normalize(d))
	{}
    __device__ float3 getPoint(float t) const { return float3_add(origin, float3_mult(t, direction)); }
	__device__ void ShiftStart(float shiftby = 1e-6) { origin = float3_add(origin,float3_mult(shiftby,direction)); }
	float3 origin, direction;
};

#endif
