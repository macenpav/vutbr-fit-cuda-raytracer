#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "phong.h"

#include "mathematics.h"

#include "ray.h"

#define EPSILON 0.000001  


struct Triangle
{
	float3	a,b,c;
	uint32	materialId;
	float3 normal;
	float3 u, v;
	uint32	id;

	__host__ __device__
	Triangle()
	{}

	__host__
	Triangle(float3 const& na, float3 const& nb, float3 const& nc, uint32 const& matId) : materialId(matId), a(na), b(nb), c(nb), u(nb - na), v(nc - na)
	{		
		normal.x = u.y*v.z - u.z*v.y;
		normal.y = u.z*v.x - u.x*v.z;
		normal.z = u.x*v.y - u.y*v.x;
		normal = normalize(normal);
	}	
	//dle http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf a
	// http://www.lighthouse3d.com/tutorials/maths/ray-triangle-intersection/
	__device__ HitInfo intersect(Ray const& ray) {

		float3 h, s, q;
		float x, f, y, z;


		h = CUDA::cross(ray.direction, v);
		x = CUDA::dot(u, h);
		HitInfo hit;
		hit.hit = false;
		if (x > -EPSILON && x < EPSILON){
			return hit;
		}

		f = 1 / x;

		s = CUDA::float3_sub(ray.origin, a);
		y = f * CUDA::dot(s, h);
		if (y < 0.0 || y > 1.0){
			return hit;
		}
		q = CUDA::cross(s, u);
		z = f * CUDA::dot(ray.direction, q);

		if (z < 0.0 || y + z > 1.0)
		{
			return hit;
		}

		hit.t = f * CUDA::dot(v, q);
		if (hit.t  > EPSILON) {
			hit.hit = true;
			
		};
		return hit;
	}
	
};


#endif
