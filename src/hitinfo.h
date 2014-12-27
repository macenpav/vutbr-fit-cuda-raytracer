#ifndef HITINFO_H
#define HITINFO_H

/**
* @brief	Declares struct holding information about ray hit.
* @author	Pavel Macenauer, Jan Bures
*/
struct HitInfo {
	/** @brief	Ray hit its target. */
	bool hit = false;
	/** @brief	Distance ray traveled to hit its target. */
	float t;
	/** @brief	Point of hit. */
	float3 point;
	/** @brief	Normal vector in the point of hit. */
	float3 normal;
	/** @brief	Material of the hit target. */
	uint32 materialId;
	/** @brief	Sphere hit by ray. */
	uint32 sphereId;
};

#endif
