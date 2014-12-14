#ifndef PHONG_H
#define PHONG_H

#include "color.h"

struct PhongMaterial
{
	void set(const Color &diff, const Color &spec, const Color &amb, float shin, float ref = 0.0)
	{ diffuse = diff; specular = spec; ambient = amb; shininess = shin; reflectance = ref; }
	
	Color diffuse;
	Color specular;
	Color ambient;
	float shininess;
	float reflectance;  
};

struct HitInfo {
	bool hit = false;
	float t;
	float3 point;
	float3 normal;
	uint32 materialId;
	uint32 sphereId;
};



#endif