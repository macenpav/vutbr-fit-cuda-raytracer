#ifndef KDTREE_H
#define KDTREE_H

#include <vector>

#include "boundingbox.h"
#include "sphere.h"
#include "constants.h"
#include "ray.h"

namespace CUDA {
	struct KDNode {
		BoundingBox boundingBox;

		KDNode* left;
		KDNode* right;

		Sphere spheres[SPLIT_LIMIT];
		uint32 countSpheres = 0;

		__device__ bool intersect(Ray const& ray)
		{
			float3 dirfrac;
			// r.dir is unit direction vector of ray
			dirfrac.x = 1.0f / ray.direction.x;
			dirfrac.y = 1.0f / ray.direction.y;
			dirfrac.z = 1.0f / ray.direction.z;
			// boundingBox.min is the corner of AABB with minimal coordinates - left bottom, boundingBox.max is maximal corner
			// ray.origin is origin of ray
			float t1 = (boundingBox.min.x - ray.origin.x)*dirfrac.x;
			float t2 = (boundingBox.max.x - ray.origin.x)*dirfrac.x;
			float t3 = (boundingBox.min.y - ray.origin.y)*dirfrac.y;
			float t4 = (boundingBox.max.y - ray.origin.y)*dirfrac.y;
			float t5 = (boundingBox.min.z - ray.origin.z)*dirfrac.z;
			float t6 = (boundingBox.max.z - ray.origin.z)*dirfrac.z;

			float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
			float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

			// if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us
			if (tmax < 0)
				return false;

			// if tmin > tmax, ray doesn't intersect AABB
			if (tmin > tmax)
				return false;

			return true;
		}
	};
}

namespace CPU {
	struct KDNode {
		BoundingBox boundingBox;

		KDNode* left;
		KDNode* right;

		std::vector<Sphere> spheres;
	};
}

inline CPU::KDNode* buildKDTree(std::vector<Sphere> const& spheres, int depth = 0) {
	CPU::KDNode* node = new CPU::KDNode;

	// copy spheres and set the pointers to null
	node->spheres = spheres;
	node->left = nullptr;
	node->right = nullptr;
	
	// calc the bounding box
	if (spheres.size()) {
		node->boundingBox = calcBoundingBox(&spheres[0]);
		for (int i = 1; i < spheres.size(); ++i) {
			expandBoundingBox(node->boundingBox, &spheres[i]);
		}
	}		

	// when there are less spheres than the limit, return node
	if (spheres.size() < SPLIT_LIMIT) {				
		return node;
	}

	// otherwise divide space	
	float3 mid_point;
	mid_point.x = 0.f;
	mid_point.y = 0.f;
	mid_point.z = 0.f;
	
	for (std::vector<Sphere>::const_iterator it = spheres.begin(); it != spheres.end(); ++it) {
		mid_point.x += it->center.x * (1.f / spheres.size());
		mid_point.y += it->center.y * (1.f / spheres.size());
		mid_point.z += it->center.z * (1.f / spheres.size());
	}

	std::vector<Sphere> left_nodes;
	std::vector<Sphere> right_nodes;
	for (std::vector<Sphere>::const_iterator it = spheres.begin(); it != spheres.end(); ++it)
	{
		switch (depth % 3)
		{
			case 0:
				if (mid_point.x < it->center.x)
					left_nodes.push_back(*it);
				else
					right_nodes.push_back(*it);
				break;
			case 1:
				if (mid_point.y < it->center.y)
					left_nodes.push_back(*it);
				else
					right_nodes.push_back(*it);
				break;
			case 2:
				if (mid_point.z < it->center.z)
					left_nodes.push_back(*it);
				else
					right_nodes.push_back(*it);
				break;
		}
	}
		
	node->left = buildKDTree(left_nodes, depth + 1);
	node->right = buildKDTree(right_nodes, depth + 1);
	
	return node;
}




inline CUDA::KDNode* copyKDTreeToDevice(CPU::KDNode* cpuNode)
{
	// address on GPU	
	CUDA::KDNode* gpuNode;
	cudaMalloc((void***)&gpuNode, sizeof(CUDA::KDNode));

	// copy CPU structure to GPU structure
	CUDA::KDNode gpuNode_data, gpuNode_data2;
	gpuNode_data.boundingBox = cpuNode->boundingBox;
	gpuNode_data.left = nullptr;
	gpuNode_data.right = nullptr;
	
	if (!cpuNode->left && !cpuNode->right)
	{
		for (uint32 i = 0; i < cpuNode->spheres.size(); ++i) {
			gpuNode_data.spheres[i] = cpuNode->spheres[i];
		}
		gpuNode_data.countSpheres = cpuNode->spheres.size();		
	}

	// recurse
	if (cpuNode->left)
		gpuNode_data.left = copyKDTreeToDevice(cpuNode->left);
	if (cpuNode->right)
		gpuNode_data.right = copyKDTreeToDevice(cpuNode->right);	

	cudaMemcpy(gpuNode, &gpuNode_data, sizeof(CUDA::KDNode), cudaMemcpyHostToDevice);
	// when we know the addresses of prev and next on GPU we can finally copy to GPU	
#ifdef _DEBUG	
	cudaMemcpy(&gpuNode_data2, gpuNode, sizeof(CUDA::KDNode), cudaMemcpyDeviceToHost);
	printf("node on GPU: min[%f,%f,%f], max[%f,%f,%f]\n", gpuNode_data2.boundingBox.min.x, gpuNode_data2.boundingBox.min.y, gpuNode_data2.boundingBox.min.z,
		gpuNode_data2.boundingBox.max.x, gpuNode_data2.boundingBox.max.y, gpuNode_data2.boundingBox.max.z);
#endif

	return gpuNode;
}

#endif
