#ifndef LIGHT_H
#define LIGHT_H

struct PointLight
{	  
	void set(float3 const& p, Color const& c)
	{ position = p; color = c; }
	
	__device__ Ray getShadowRay(float3 const& point) { return Ray(point, CUDA::float3_sub(position, point)); }		
	
	float3 position;
	Color color;
};

#endif