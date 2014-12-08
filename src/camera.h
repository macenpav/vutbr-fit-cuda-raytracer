#ifndef CAMERA_H
#define CAMERA_H

struct Camera {

	void init()
	{		 
		position = make_float3(0.f, 0.f, 0.f);
		direction = make_float3(0.f, 0.f, 1.f);
		right = make_float3(1.f, 0.f, 0.f);
		up = make_float3(0.f, 1.f, 0.f);
	}

	void lookAt(float3 const& eye, float3 const& target, float3 const& sky, float const& fov, float const& ratio) {
		position = eye;	

		direction = target - eye;
		normalize(direction);

		right = cross(sky, direction);
		normalize(right);
		right *= ratio;
		
		up = cross(direction, right);
		normalize(up);		
	}

	__device__ Ray getRay(float u, float v) const {
		return Ray(position, CUDA::float3_add(direction, CUDA::float3_mult(u, right), CUDA::float3_mult(v, up)));
	}

	float3 position;
	float3 direction;
	float3 right;
	float3 up;

};


#endif
