#ifndef LIGHT_H
#define LIGHT_H

/** 
 * @brief	Declares a point light type. 
 * @author	Pavel Macenauer, Jan Bures 
 */
struct PointLight
{	
	__host__ __device__
	PointLight() { }

	__host__ __device__
	PointLight(float3 const& p, Color const& c)
	{ position = p; color = c; }
	
	__device__ Ray getShadowRay(float3 const& point)
	{ return Ray(point, float3_sub(position, point)); }		
	
	/** @brief	Light position. */
	float3 position;
	/** @brief	Light color. */
	Color color;
};

#endif
