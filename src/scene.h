#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "sphere.h"
#include "plane.h"
#include "camera.h"
#include "light.h"

struct SceneStats{
	uint32 sphereCount;
	uint32 planeCount;
	uint32 lightCount;
};

class Scene
{
public: 
	uint32 getSphereCount() const { return spheres.size(); }
	uint32 getPlaneCount() const { return planes.size(); }
	uint32 getLightCount() const { return lights.size(); }

	void add(Sphere s) { spheres.push_back(s); }
	void add(Plane p){ planes.push_back(p); }
	void add(PointLight p){ lights.push_back(p); }
	void add(PhongMaterial mat) { materials.push_back(mat); }

	Camera* getCamera() { return &camera; }
	Sphere* getSpheres() { return &spheres[0]; }
	Plane* getPlanes() { return &planes[0];	}
	PointLight* getLights() { return &lights[0]; }
	PhongMaterial* getMaterials() { return &materials[0]; }

	SceneStats* getSceneStats(){
		sceneStats.planeCount = planes.size();
		sceneStats.sphereCount = spheres.size();
		sceneStats.lightCount = lights.size();
		return &sceneStats;
	};

	std::vector<Sphere> getSphereVector() const { return spheres; }

private:
	std::vector<PhongMaterial> materials;
	std::vector<Sphere> spheres;
	std::vector<Plane> planes;
	std::vector<PointLight> lights;
	Camera camera;
	SceneStats sceneStats;
};

#endif