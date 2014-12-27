#ifndef PHONG_H
#define PHONG_H

#include "color.h"

/**
* @brief	Declares the Phong material struct.
* @author	Pavel Macenauer, Jan Bures
*/
struct PhongMaterial
{
	__device__ __host__
	PhongMaterial(){ };

	__device__ __host__
	PhongMaterial(const Color &diff, const Color &spec, const Color &amb, float shin, float ref = 0.0)
	{ diffuse = diff; specular = spec; ambient = amb; shininess = shin; reflectance = ref; }

	__device__ __host__
	void set(const Color &diff, const Color &spec, const Color &amb, float shin, float ref = 0.0)
	{ diffuse = diff; specular = spec; ambient = amb; shininess = shin; reflectance = ref; }
	
	/** @brief	Diffuse light. */
	Color diffuse;
	/** @brief	Specular light. */
	Color specular;
	/** @brief	Ambient light. */
	Color ambient;
	/** @brief	Shininess. */
	float shininess;
	/** @brief	Reflectance. */
	float reflectance;  
};

const PhongMaterial MAT_RED(COLOR_RED, COLOR_WHITE, Color(0.1f, 0.05f, 0.05f), DEFAULT_PHONG_SHININESS);
const PhongMaterial MAT_RED_REFL(COLOR_RED, COLOR_WHITE, Color(0.1f, 0.05f, 0.05f), DEFAULT_PHONG_SHININESS, DEFAULT_PHONG_REFLECTANCE);
const PhongMaterial MAT_GREEN(COLOR_GREEN, COLOR_WHITE, Color(0.25f, 0.f, 0.f), DEFAULT_PHONG_SHININESS);
const PhongMaterial MAT_GREEN_REFL(COLOR_GREEN, COLOR_WHITE, Color(0.25f, 0.f, 0.f), DEFAULT_PHONG_SHININESS, DEFAULT_PHONG_REFLECTANCE);
const PhongMaterial MAT_BLUE(COLOR_BLUE, COLOR_WHITE, Color(0.15f, 0.1f, 0.1f), DEFAULT_PHONG_SHININESS);
const PhongMaterial MAT_BLUE_REFL(COLOR_BLUE, COLOR_WHITE, Color(0.15f, 0.1f, 0.1f), DEFAULT_PHONG_SHININESS, DEFAULT_PHONG_REFLECTANCE);
const PhongMaterial MAT_WHITE(COLOR_LIGHTGRAY, COLOR_WHITE, Color(0.15f, 0.15f, 0.15f), DEFAULT_PHONG_SHININESS);
const PhongMaterial MAT_BLACK(COLOR_BLACK, COLOR_WHITE, Color(0.f, 0.f, 0.f), DEFAULT_PHONG_SHININESS);
const PhongMaterial MAT_MIRROR(COLOR_LIGHTGRAY, COLOR_WHITE, Color(0.15f, 0.15f, 0.15f), DEFAULT_PHONG_SHININESS, DEFAULT_PHONG_REFLECTANCE);

#endif
