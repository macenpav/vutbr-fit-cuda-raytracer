#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include <vector_types.h>
#include <cmath>

#include "sphere.h"


struct BoundingBox {
	float3 min;
	float3 max;	
};

inline BoundingBox calcBoundingBox(const Sphere* sphere) {
	BoundingBox bbox;

	bbox.min.x = sphere->center.x - sphere->radius;
	bbox.min.y = sphere->center.y - sphere->radius;
	bbox.min.z = sphere->center.z - sphere->radius;

	bbox.max.x = sphere->center.x + sphere->radius;
	bbox.max.y = sphere->center.y + sphere->radius;
	bbox.max.z = sphere->center.z + sphere->radius;

	return bbox;
}

inline void expandBoundingBox(BoundingBox& bbox, const Sphere* sphere) {
	bbox.min.x = std::fmin(bbox.min.x, sphere->center.x - sphere->radius);
	bbox.min.y = std::fmin(bbox.min.y, sphere->center.y - sphere->radius);
	bbox.min.z = std::fmin(bbox.min.z, sphere->center.z - sphere->radius);

	bbox.max.x = std::fmax(bbox.max.x, sphere->center.x + sphere->radius);
	bbox.max.y = std::fmax(bbox.max.y, sphere->center.y + sphere->radius);
	bbox.max.z = std::fmax(bbox.max.z, sphere->center.z + sphere->radius);
}

#endif
