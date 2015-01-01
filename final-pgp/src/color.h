#ifndef COLOR_H
#define COLOR_H

#include "mathematics.h"

/**
* @brief	Declares the RGB color struct.
* @author	Pavel Macenauer, Jan Bures
*/
struct Color
{			
	__host__ __device__ 
	Color() {}

	__host__ __device__ 
	Color(float r, float g, float b) { red = r; green = g; blue = b; }

	__host__ __device__ 
	void set(float const& r, float const& g, float const& b) { red = r; green = g; blue = b; }
	
	__host__ __device__ 
	void accumulate(Color const& x, float const& scale = 1.0) { red += scale*x.red; green += scale*x.green; blue += scale*x.blue; }

	__host__ __device__ 
	Color &operator *= (float const& x) { red *= x; green *= x; blue *= x; return *this; }

	__host__ __device__ 
	Color operator + (Color const& x) { Color c; c.red = this->red + x.red; c.green = this->green + x.green; c.blue = this->blue + x.blue; return c; };

	__host__ __device__
	Color operator - (Color const& x) { Color c; c.red = this->red - x.red; c.green = this->green - x.green; c.blue = this->blue - x.blue; return c; };

	__host__ __device__
	Color operator * (Color const& x) { Color c; c.red = this->red * x.red; c.green = this->green * x.green; c.blue = this->blue * x.blue; return c; };

	__host__ __device__
	Color operator / (Color const& x) { Color c; c.red = this->red / x.red; c.green = this->green / x.green; c.blue = this->blue / x.blue; return c; };

	/** @brief	Red property. */
	float red;
	/** @brief	Green property. */
	float green;
	/** @brief	Blue property. */
	float blue;
};

const Color COLOR_RED(1.f, 0.f, 0.f);
const Color COLOR_GREEN(0.f, 1.f, 0.f);
const Color COLOR_BLUE(0.f, 0.f, 1.f);
const Color COLOR_WHITE(1.f, 1.f, 1.f);
const Color COLOR_BLACK(0.f, 0.f, 0.f);
const Color COLOR_LIGHTGRAY(0.75f, 0.75f, 0.75f);
const Color COLOR_DARKGRAY(0.15f, 0.15f, 0.15f);
const Color COLOR_YELLOW(1.f, 1.f, 0.f);
const Color COLOR_MAGENTA(1.f, 0.f, 1.f);
const Color COLOR_CYAN(0.f, 1.f, 1.f);

#endif
